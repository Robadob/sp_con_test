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
		float histogram = 0;
		float scan = 0;
		float reorder = 0;
		float tex = 0;
	};

	namespace
	{
		__global__ void atomicHistogram(unsigned int* bin_index, unsigned int* bin_sub_index, unsigned int *pbm_counts, glm::vec2 *messageBuffer)
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
			unsigned int* bin_index,
			unsigned int* bin_sub_index,
			unsigned int *pbm,
			glm::vec2 *unordered_messages,
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
		Times construct()
		{
			Times t = {};

			cudaEvent_t start_PBM, end_histogram, end_scan, end_reorder, end_PBM;
			cudaEventCreate(&start_PBM);
			cudaEventCreate(&end_histogram);
			cudaEventCreate(&end_scan);
			cudaEventCreate(&end_reorder);
			cudaEventCreate(&end_PBM);
			//BuildPBM
			unsigned int *d_PBM_counts = nullptr;
			unsigned int *d_PBM = nullptr;
			CUDA_CALL(cudaMalloc(&d_PBM_counts, (BIN_COUNT + 1) * sizeof(unsigned int)));
			CUDA_CALL(cudaMalloc(&d_PBM, (BIN_COUNT + 1) * sizeof(unsigned int)));
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
			float pbmMillis = 0, kernelMillis = 0;
			const unsigned int ITERATIONS = 200;
			for (unsigned int i = 0; i < ITERATIONS; ++i)
			{
				//Reset each run of average model
#ifndef CIRCLES
				CUDA_CALL(cudaMemcpy(d_out, d_agents_init, sizeof(glm::vec2)*AGENT_COUNT, cudaMemcpyDeviceToDevice));
#endif
				cudaEventRecord(start_PBM);
				{//Build atomic histogram
					CUDA_CALL(cudaMemset(d_PBM_counts, 0x00000000, (BIN_COUNT + 1) * sizeof(unsigned int)));
					int blockSize;   // The launch configurator returned block size 
					CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram, 32, 0));//Randomly 32
																												 // Round up according to array size
					int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
					atomicHistogram << <gridSize, blockSize >> >(d_keys, d_vals, d_PBM_counts, d_out);
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
					int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
					//Copy messages from d_messages to d_messages_swap, in hash order
					reorderLocationMessages << <gridSize, blockSize >> >(d_keys, d_vals, d_PBM, d_out, d_agents);
					CUDA_CHECK();
					//Wait for return
					CUDA_CALL(cudaDeviceSynchronize());
				}
				cudaEventRecord(end_reorder);
				{//Fill PBM and Message Texture Buffers
					CUDA_CALL(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec2) * AGENT_COUNT));
					CUDA_CALL(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (BIN_COUNT + 1)));
				}
				cudaEventRecord(end_PBM);
				CUDA_CALL(cudaDeviceSynchronize());
				//Accumulate timings
				cudaEventSynchronize(end_PBM);
				Times _t;
				cudaEventElapsedTime(&_t.overall, start_PBM, end_histogram);
				cudaEventElapsedTime(&_t.histogram, start_PBM, end_PBM);
				cudaEventElapsedTime(&_t.scan, end_histogram, end_scan);
				cudaEventElapsedTime(&_t.reorder, end_scan, end_reorder);
				cudaEventElapsedTime(&_t.tex, end_reorder, end_PBM);
				t.overall += _t.overall;
				t.histogram += _t.histogram;
				t.scan += _t.scan;
				t.reorder += _t.reorder;
				t.tex += _t.tex;
			}
			//Reduce to average
			t.overall /= ITERATIONS;
			t.histogram /= ITERATIONS;
			t.scan /= ITERATIONS;
			t.reorder /= ITERATIONS;
			t.tex /= ITERATIONS;

			return t;
		}
	}
}
#endif //__atomic_cuh__