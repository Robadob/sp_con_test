#ifndef __original_cuh__
#define __original_cuh__
#include "common.h"
/**
 * Original, non-atomic method of constructing USP data-structure
 */
namespace original
{
	struct Times : ConstructionTimes
	{
		float hash = 0;
		float sort = 0;
		float memset = 0;
		float reorder = 0;
		float tex = 0;
	};
	namespace
	{
		__global__ void hashLocationMessages(unsigned int* keys, unsigned int* vals, const glm::vec2* messageBuffer)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			//Kill excess threads
			if (index >= d_agentCount) return;

			glm::uvec2 gridPos = getGridPosition(messageBuffer[index]);
			unsigned int hash = getHash(gridPos);
			keys[index] = hash;
			vals[index] = index;
		}
		__global__ void reorderLocationMessages2(
			const unsigned int *keys,
			const unsigned int *vals,
			unsigned int *pbm_index,
			unsigned int *pbm_count,
			const glm::vec2 *unordered_messages,
			glm::vec2 *ordered_messages
		)
		{
			extern __shared__ int sm_data[];

			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			////Load current key and copy it into shared
			unsigned int key;
			unsigned int old_pos;
			if (index < d_agentCount)
			{//Don't go out of bounds when buffer is at max capacity
				key = keys[index];
				old_pos = vals[index];
				//Every valid thread put key into shared memory
				sm_data[threadIdx.x] = key;
			}
			__syncthreads();
			////Kill excess threads
			if (index >= d_agentCount) return;

			//Load next key
			unsigned int prev_key;
			//if thread is final thread
			if (index == 0)
			{
				prev_key = UINT_MAX;//?
			}
			//If thread is first in block, no next in SM, goto global
			else if (threadIdx.x == 0)
			{
				prev_key = keys[index - 1];
			}
			else
			{
				prev_key = sm_data[threadIdx.x - 1];
			}
			//Boundary message
			if (prev_key != key)
			{
				pbm_index[key] = index;
				if (index > 0)
				{
					pbm_count[key - 1] = index;
				}
			}
			if (index == d_agentCount - 1)//New debugger stalls here.
			{
				pbm_count[key] = d_agentCount;
			}
#ifdef _DEBUG
			if (old_pos >= d_agentCount)
			{
				printf("ERROR: PBM generated an out of range old_pos (%i >= %i).\n", old_pos, d_agentCount);
				assert(0);
			}
#endif


			//Order messages into swap space
			ordered_messages[index] = unordered_messages[old_pos];
			
#ifdef _DEBUG
			//Check these rather than ordered in hopes of memory coealesce
			if (ordered_messages[index].x == NAN
				|| ordered_messages[index].y == NAN
				)
			{
				printf("ERROR: Location containing NaN detected.\n");
			}
#endif
		}
		int requiredSM_reorderLocationMessages(int blockSize)
		{
			return sizeof(unsigned int)*blockSize;
		}
		size_t d_CUB_temp_storage_bytes = 0;
		void *d_CUB_temp_storage = nullptr;
	}
	Times construct(const unsigned int &POPULATION_SIZE, const glm::uvec2 &DIMS, const unsigned int &BIN_COUNT)
	{
		const unsigned long BIN_COUNT_BITS = (unsigned long)ceil(log(BIN_COUNT) / log(2));

		Times t = {};

		cudaEvent_t start_PBM, end_hash, end_sort, end_memset, end_reorder, end_PBM;
		cudaEventCreate(&start_PBM);
		cudaEventCreate(&end_hash);
		cudaEventCreate(&end_sort);
		cudaEventCreate(&end_memset);
		cudaEventCreate(&end_reorder);
		cudaEventCreate(&end_PBM);
		{//Resize cub temp if required
			size_t bytesCheck;
			cub::DeviceRadixSort::SortPairs(nullptr, bytesCheck, d_keys, d_keys_swap, d_vals, d_vals_swap, POPULATION_SIZE, 0, BIN_COUNT_BITS);
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
		cudaEventRecord(start_PBM);
		//Fill primitive key/val arrays for sort
		{
			int blockSize;   // The launch configurator returned block size 
			CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, hashLocationMessages, 32, 0));//Randomly 32
		    // Round up according to array size
			int gridSize = (POPULATION_SIZE + blockSize - 1) / blockSize;
			hashLocationMessages << <gridSize, blockSize >> >(d_keys, d_vals, d_agents_in);
			CUDA_CALL(cudaDeviceSynchronize());
		}
		cudaEventRecord(end_hash);
		//Sort key val arrays using thrust/CUB
		{
			cub::DeviceRadixSort::SortPairs(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_keys, d_keys_swap, d_vals, d_vals_swap, POPULATION_SIZE, 0, BIN_COUNT_BITS);
			//Swap arrays
			unsigned int *temp;
			temp = d_keys;
			d_keys = d_keys_swap;
			d_keys_swap = temp;
			temp = d_vals;
			d_vals = d_vals_swap;
			d_vals_swap = temp;
			////Free temporary memory
			//cudaFree(d_temp_storage);
			CUDA_CALL(cudaGetLastError());
		}
		cudaEventRecord(end_sort);
		//Reorder map in order of message_hash	
		//Fill pbm start coords with known value 0xffffffff
		//CUDA_CALL(cudaMemset(d_PBM, 0xffffffff, BIN_COUNT * sizeof(int)));
		//Fill pbm end coords with known value 0x00000000 (this should mean if the mysterious bug does occur, the cell is just dropped, not large loop created)
		CUDA_CALL(cudaMemset(d_PBM, 0xffffffff, BIN_COUNT * sizeof(unsigned int)));

		cudaEventRecord(end_memset);
		{//Reorder messages and create PBM index
			int minGridSize, blockSize;   // The launch configurator returned block size 
			cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, reorderLocationMessages2, requiredSM_reorderLocationMessages, 0);
			// Round up according to array size
			int gridSize = (POPULATION_SIZE + blockSize - 1) / blockSize;
			//Copy messages from d_messages to d_messages_swap, in hash order
			reorderLocationMessages2 << <gridSize, blockSize, requiredSM_reorderLocationMessages(blockSize) >> >(d_keys, d_vals, d_PBM, d_PBM_counts, d_agents_in, d_agents_out);
			CUDA_CHECK();
			//Wait for return
			CUDA_CALL(cudaDeviceSynchronize());
		}
		cudaEventRecord(end_reorder);
		{//Fill PBM and Message Texture Buffers
			CUDA_CALL(cudaBindTexture(nullptr, d_texMessages, d_agents_out, sizeof(glm::vec2) * POPULATION_SIZE));
			CUDA_CALL(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (BIN_COUNT)));
			CUDA_CALL(cudaBindTexture(nullptr, d_texPBM_counts, d_PBM_counts, sizeof(unsigned int) * (BIN_COUNT)));
		}
		cudaEventRecord(end_PBM);
		CUDA_CALL(cudaDeviceSynchronize());
		//Release temp resources
		CUDA_CALL(cudaUnbindTexture(d_texPBM_counts));
		CUDA_CALL(cudaUnbindTexture(d_texPBM));
		CUDA_CALL(cudaUnbindTexture(d_texMessages));
		//Accumulate timings
		cudaEventSynchronize(end_PBM);
		cudaEventElapsedTime(&t.overall, start_PBM, end_PBM);
		cudaEventElapsedTime(&t.hash, start_PBM, end_hash);
		cudaEventElapsedTime(&t.sort, end_hash, end_sort);
		cudaEventElapsedTime(&t.memset, end_sort, end_memset);
		cudaEventElapsedTime(&t.reorder, end_memset, end_reorder);
		cudaEventElapsedTime(&t.tex, end_reorder, end_PBM);
		CUDA_CHECK();
		return t;
	}
	void logHeader(std::ofstream &f, unsigned int &i)
	{
		f << "(" << (i++) << ") " << "Original_Overall,";
		f << "(" << (i++) << ") " << "Original_Hash,";
		f << "(" << (i++) << ") " << "Original_Sort,";
		f << "(" << (i++) << ") " << "Original_Memset,";
		f << "(" << (i++) << ") " << "Original_Reorder,";
		f << "(" << (i++) << ") " << "Original_Tex,";
	}
	void logResult(std::ofstream &f, const Times &t)
	{
		f << t.overall << ",";
		f << t.hash << ",";
		f << t.sort << ",";
		f << t.memset << ",";
		f << t.reorder << ",";
		f << t.tex << ",";
	}
}
#endif //__original_cuh__