#ifndef __common_h__
#define __common_h__

#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <glm/glm.hpp>
#include <curand_kernel.h>
#include <texture_fetch_functions.h>

#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/constants.hpp>
#include <fstream>
#include <string>
#include <ctime>

//Common super-struct for timings
struct ConstructionTimes
{
	float overall = 0;
};
//CUDA error handling
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

//Cuda Constant
__device__ __constant__ unsigned int d_agentCount;
__device__ __constant__ glm::uvec2 d_gridDim;
__device__ __constant__ glm::vec2 d_gridDim_float;
__device__ __constant__ float d_RADIUS;
__device__ __constant__ float d_R_SIN_45;
//Dynamic cuda memory
glm::vec2 *d_agents_in = nullptr;
glm::vec2 *d_agents_out = nullptr;
//Common
unsigned int *d_keys = nullptr;
unsigned int *d_vals = nullptr;
unsigned int *d_PBM_counts = nullptr;
unsigned int *d_PBM = nullptr;
//original.cuh
unsigned int *d_keys_swap = nullptr;
unsigned int *d_vals_swap = nullptr;

//Tex buffers
texture<float2> d_texMessages;
texture<unsigned int> d_texPBM;
//Double array PBM from original.cuh
texture<unsigned int> d_texPBM_counts;

const unsigned long long RNG_SEED = 12;
//Common util functions
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
	locationMessages[id].x = (-curand_uniform(&state[id]) + 1.0f)*d_gridDim_float.x;
	locationMessages[id].y = (-curand_uniform(&state[id]) + 1.0f)*d_gridDim_float.y;
}
__device__ __forceinline__ glm::ivec2 getGridPosition(glm::vec2 worldPos)
{
	//Clamp each grid coord to 0<=x<dim
	return clamp(floor(worldPos), glm::vec2(0), glm::vec2(d_gridDim) - glm::vec2(1.0f));
}
__device__ __forceinline__ unsigned int getHash(glm::ivec2 gridPos)
{
	//Bound gridPos to gridDimensions
	gridPos = clamp(gridPos, glm::ivec2(0), glm::ivec2(d_gridDim) - glm::ivec2(1));
	//Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
	return (unsigned int)(
		(gridPos.y * d_gridDim.x) +					//y
		gridPos.x); 	                            //x
}
#endif //__common_h__