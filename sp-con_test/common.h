#ifndef __common_h__
#define __common_h__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <glm/glm.hpp>
#include <curand_kernel.h>
#include <texture_fetch_functions.h>

struct ConstructionTimes
{
	float overall = 0;
};

static void HandleCUDAError(const char *file,
	int line,
	cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
	cudaDeviceSynchronize();
#endif
	if (status != cudaError::cudaSuccess || (status = cudaGetLastError()) != cudaError::cudaSuccess)
	{
		printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line, cudaGetErrorString(status));
#ifdef _DEBUG
		getchar();
#endif
		exit(1);
	}
}
#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))
#define CUDA_CHECK() (HandleCUDAError(__FILE__, __LINE__))

__device__ __constant__ unsigned int d_agentCount;
__device__ __constant__ float d_environmentWidth_float;
__device__ __constant__ unsigned int d_gridDim;
__device__ __constant__ float d_gridDim_float;
__device__ __constant__ float d_RADIUS;
__device__ __constant__ float d_R_SIN_45;
__device__ __constant__ float d_binWidth;

texture<float2> d_texMessages;
texture<unsigned int> d_texPBM;

__global__ void init_curand(curandState *state, unsigned long long seed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < d_agentCount)
		curand_init(seed, id, 0, &state[id]);
}
__global__ void init_agents(curandState *state, glm::vec2 *locationMessages) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= d_agentCount)
		return;
	//curand_unform returns 0<x<=1.0, not much can really do about 0 exclusive
	//negate and  + 1.0, to make  0<=x<1.0
	locationMessages[id].x = (-curand_uniform(&state[id]) + 1.0f)*d_environmentWidth_float;
	locationMessages[id].y = (-curand_uniform(&state[id]) + 1.0f)*d_environmentWidth_float;
}
__device__ __forceinline__ glm::ivec2 getGridPosition(glm::vec2 worldPos)
{
	//Clamp each grid coord to 0<=x<dim
	return clamp(floor((worldPos / d_environmentWidth_float)*d_gridDim_float), glm::vec2(0), glm::vec2((float)d_gridDim - 1));
}
__device__ __forceinline__ unsigned int getHash(glm::ivec2 gridPos)
{
	//Bound gridPos to gridDimensions
	gridPos = clamp(gridPos, glm::ivec2(0), glm::ivec2(d_gridDim - 1));
	//Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
	return (unsigned int)(
		(gridPos.y * d_gridDim) +					//y
		gridPos.x); 	                            //x
}
#endif //__common_h__