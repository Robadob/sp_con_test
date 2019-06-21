#ifndef __atomic_cuh__
#define __atomic_cuh__
#include "common.h"
/**
 * Atomic method of constructing USP data-structure
 */
//Based on: https://github.com/Robadob/sp-2018-08/blob/master/2D/kernel2D.cu
namespace atomic
{
	struct Times : ConstructionTimes
	{
		float memset = 0;
		float histogram = 0;
		float scan = 0;
		float reorder = 0;
		float tex = 0;
	};

	namespace
	{
		__global__ void atomicHistogram(unsigned int* bin_index, unsigned int* bin_sub_index, unsigned int *pbm_counts, const glm::vec2 *messageBuffer)
		{
			unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			//Kill excess threads
			if (index >= d_agentCount) return;

			glm::ivec2 gridPos = getGridPosition(messageBuffer[index]);
			unsigned int hash = getHash(gridPos);
			bin_index[index] = hash;
			unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
			bin_sub_index[index] = bin_idx;
		}
		__global__ void reorderLocationMessages(
			const unsigned int* bin_index,
			const unsigned int* bin_sub_index,
			const unsigned int *pbm,
			const glm::vec2 *unordered_messages,
			glm::vec2 *ordered_messages
		)
		{
			unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			//Kill excess threads
			if (index >= d_agentCount) return;

			unsigned int i = bin_index[index];
			unsigned int sorted_index = pbm[i] + bin_sub_index[index];

			//Order messages into swap space
			ordered_messages[sorted_index] = unordered_messages[index];
		}
		size_t d_CUB_temp_storage_bytes = 0;
		void *d_CUB_temp_storage = nullptr;
	}
	Times construct(const unsigned int &POPULATION_SIZE, const glm::uvec2 &DIMS, const unsigned int &BIN_COUNT)
	{
		Times t = {};

		cudaEvent_t start_PBM, end_memset, end_histogram, end_scan, end_reorder, end_PBM;
		cudaEventCreate(&start_PBM);
		cudaEventCreate(&end_memset);
		cudaEventCreate(&end_histogram);
		cudaEventCreate(&end_scan);
		cudaEventCreate(&end_reorder);
		cudaEventCreate(&end_PBM);
		{//Resize cub temp if required
			size_t bytesCheck;
			cub::DeviceScan::ExclusiveSum(nullptr, bytesCheck, d_PBM, d_PBM_counts, BIN_COUNT + 1);
			if (bytesCheck > d_CUB_temp_storage_bytes)
			{
				if (d_CUB_temp_storage)
				{
					CUDA_CALL(cudaFree(d_CUB_temp_storage));
				}
				d_CUB_temp_storage_bytes = bytesCheck;
				CUDA_CALL(cudaMalloc(&d_CUB_temp_storage, d_CUB_temp_storage_bytes));
			}
		}

		//For 200 iterations (to produce an average)
		const unsigned int ITERATIONS = 1;
		//for (unsigned int i = 0; i < ITERATIONS; ++i)
		//{
		//	//Reset each run of average model
		//	CUDA_CALL(cudaMemcpy(d_out, d_agents_init, sizeof(glm::vec2)*POPULATION_SIZE, cudaMemcpyDeviceToDevice));

			cudaEventRecord(start_PBM);
			{//Reset histogram
				CUDA_CALL(cudaMemset(d_PBM_counts, 0x00000000, (BIN_COUNT + 1) * sizeof(unsigned int)));
			}
			cudaEventRecord(end_memset);
			{//Build atomic histogram
				int blockSize;   // The launch configurator returned block size 
				CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram, 32, 0));//Randomly 32
																											 // Round up according to array size
				int gridSize = (POPULATION_SIZE + blockSize - 1) / blockSize;
				atomicHistogram <<<gridSize, blockSize >> >(d_keys, d_vals, d_PBM_counts, d_agents_in);
				CUDA_CALL(cudaDeviceSynchronize());
			}
			cudaEventRecord(end_histogram);
			{//Scan (sum), to finalise PBM
				cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_PBM_counts, d_PBM, BIN_COUNT + 1);
			}
			cudaEventRecord(end_scan);
			{//Reorder messages
				int blockSize;   // The launch configurator returned block size 
				CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
																													 // Round up according to array size
				int gridSize = (POPULATION_SIZE + blockSize - 1) / blockSize;
				//Copy messages from d_messages to d_messages_swap, in hash order
				reorderLocationMessages << <gridSize, blockSize >> >(d_keys, d_vals, d_PBM, d_agents_in, d_agents_out);
				CUDA_CHECK();
				//Wait for return
				CUDA_CALL(cudaDeviceSynchronize());
			}
			cudaEventRecord(end_reorder);
			{//Fill PBM and Message Texture Buffers
				CUDA_CALL(cudaBindTexture(nullptr, d_texMessages, d_agents_out, sizeof(glm::vec2) * POPULATION_SIZE));
				CUDA_CALL(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (BIN_COUNT + 1)));
			}
			cudaEventRecord(end_PBM);
			CUDA_CALL(cudaDeviceSynchronize());
			//Copy output to input array
			CUDA_CALL(cudaMemcpy(d_agents_in, d_agents_out, sizeof(glm::vec2)*POPULATION_SIZE, cudaMemcpyDeviceToDevice));
		    //Release temp resources
			CUDA_CALL(cudaUnbindTexture(d_texPBM));
			CUDA_CALL(cudaUnbindTexture(d_texMessages));
			//Accumulate timings
			cudaEventSynchronize(end_PBM);
			Times _t;
			cudaEventElapsedTime(&_t.overall, start_PBM, end_PBM);
			cudaEventElapsedTime(&_t.memset, start_PBM, end_memset);
			cudaEventElapsedTime(&_t.histogram, end_memset, end_histogram);
			cudaEventElapsedTime(&_t.scan, end_histogram, end_scan);
			cudaEventElapsedTime(&_t.reorder, end_scan, end_reorder);
			cudaEventElapsedTime(&_t.tex, end_reorder, end_PBM);
			t.overall += _t.overall;
			t.memset += _t.memset;
			t.histogram += _t.histogram;
			t.scan += _t.scan;
			t.reorder += _t.reorder;
			t.tex += _t.tex;
		//}//for-ITERATIONS
		//Reduce to average
		t.overall /= ITERATIONS;
		t.memset /= ITERATIONS;
		t.histogram /= ITERATIONS;
		t.scan /= ITERATIONS;
		t.reorder /= ITERATIONS;
		t.tex /= ITERATIONS;
		CUDA_CHECK();
		return t;
	}
	void logHeader(std::ofstream &f, unsigned int &i)
	{
		f << "(" << (i++) << ") " << "Atomic_Overall,";
		f << "(" << (i++) << ") " << "Atomic_Memset,";
		f << "(" << (i++) << ") " << "Atomic_Histogram,";
		f << "(" << (i++) << ") " << "Atomic_Scan,";
		f << "(" << (i++) << ") " << "Atomic_Reorder,";
		f << "(" << (i++) << ") " << "Atomic_Tex,";
	}
	void logResult(std::ofstream &f, const Times &t)
	{
		f << t.overall << ",";
		f << t.memset << ",";
		f << t.histogram << ",";
		f << t.scan << ",";
		f << t.reorder << ",";
		f << t.tex << ",";
	}
}
#endif //__atomic_cuh__