/**
 * Original vs Atomic
 * Static distribution (e.g. new bins remain empty) vs scaling distribution (uniform rng over env)
 * Scaling environment by extending 1 dimension
 */

#include <cstdio>
#include <cstdlib>

#include "original.cuh"
#include "atomic.cuh"

#define RETRIES 200

/**
 * Agent pop remains in initial distribution and environment grows around them
 * @param popSize Total number of agents which inhabit the environment.
 * @param startBins The (approximate) number of bins of the initial run
 * @param endBins The (approximate) number of bins of the final run
 * @param steps The number of steps to be executed (total runs = steps+1)
 */
void runStatic(const unsigned int &popSize, const unsigned int &startBins, const unsigned int &endBins, const unsigned int &steps, const char *logPath);
/**
 * Agent pop remains uniform randomly distributed about the environment
 * This will take a while to run for large agent pop sizes, due to cuRand.
 * @param popSize Total number of agents which inhabit the environment.
 * @param startBins The (approximate) number of bins of the initial run
 * @param endBins The (approximate) number of bins of the final run
 * @param steps The number of steps to be executed (total runs = steps+1)
 */
void runDynamic(const unsigned int &popSize, const unsigned int &startBins, const unsigned int &endBins, const unsigned int &steps, const char *logPath);

int main(int argc, char **argv)
{
	if(argc>1)
	{//Set cuda device
		long int cudaDevice = strtol(argv[1], nullptr, 0);
		printf("Attempting to set CUDA device %d.\n", cudaDevice);
		cudaSetDevice(static_cast<int>(cudaDevice));
	}
	//Static distribution of agents
	//runStatic(100000, 5000, 15000000, 200, "static-100k");
	//runStatic(1000000, 5000, 15000000, 200, "static-1m");
	//runDynamic(100000, 5000, 15000000, 200, "dynamic-100k");
	//runDynamic(1000000, 5000, 15000000, 200, "dynamic-1m");
	//runDynamic(50000, 5000, 20000000, 200, "dynamic-50k");//Most representative of how I found this with FGPU/KeratinoCyte
	runDynamic(500000, 1000, 2000000, 200, "dynamic-500k");
	return EXIT_SUCCESS;
}

void runStatic(const unsigned int &POP_SIZE, const unsigned int &START_BINS, const unsigned int &END_BINS, const unsigned int &STEPS, const char *logName)
{
	//Defined out of scope of methods being tested
	glm::vec2 *d_agents_init;
	curandState *d_rng = nullptr;
	//Pre-calc init final state
	const glm::uvec2 INIT_DIMS = glm::uvec2(static_cast<unsigned int>(floor(sqrt(START_BINS))));
	const unsigned int INIT_BINS = glm::compMul(INIT_DIMS);
	const glm::uvec2 FINAL_DIMS = glm::uvec2(INIT_DIMS.x, ceil(static_cast<float>(END_BINS)/ INIT_DIMS.x));
	const unsigned int FINAL_BINS = glm::compMul(FINAL_DIMS);
	const glm::vec2 STEP_DIMS = glm::vec2(0, static_cast<float>(FINAL_DIMS.y - INIT_DIMS.y) / STEPS);
	const unsigned int AVERAGE_NEIGHBOURS = static_cast<unsigned int>(POP_SIZE * glm::pi<float>() / INIT_BINS);
	//Allocate SP memory
	{
		//Agents
		CUDA_CALL(cudaMalloc(&d_agents_in, sizeof(glm::vec2) * POP_SIZE));//Redundant, theoretically d_agents_in should remain unchanged.
		CUDA_CALL(cudaMalloc(&d_agents_out, sizeof(glm::vec2) * POP_SIZE));
		//Agent reset
		CUDA_CALL(cudaMalloc(&d_agents_init, sizeof(glm::vec2) * POP_SIZE));
		//SP interim
		CUDA_CALL(cudaMalloc(&d_keys, sizeof(unsigned int) * POP_SIZE));
		CUDA_CALL(cudaMalloc(&d_vals, sizeof(unsigned int) * POP_SIZE));
		//Final PBM storage
		CUDA_CALL(cudaMalloc(&d_PBM_counts, (FINAL_BINS + 1) * sizeof(unsigned int)));
		CUDA_CALL(cudaMalloc(&d_PBM, (FINAL_BINS + 1) * sizeof(unsigned int)));
		//original.cuh swap space
		CUDA_CALL(cudaMalloc(&d_keys_swap, sizeof(unsigned int) * POP_SIZE));
		CUDA_CALL(cudaMalloc(&d_vals_swap, sizeof(unsigned int) * POP_SIZE));
	}
	//Init Actor Population
	{
		//Temp dims for init
		glm::vec2 dims_float = glm::vec2(INIT_DIMS);
		CUDA_CALL(cudaMemcpyToSymbol(d_gridDim_float, &dims_float, sizeof(glm::vec2)));
		//cuRand
		CUDA_CALL(cudaMalloc(&d_rng, POP_SIZE * sizeof(curandState)));
		//Arbitrary thread block sizes (speed not too important during one off initialisation)
		unsigned int initThreads = 512;
		unsigned int initBlocks = (POP_SIZE / initThreads) + 1;
		init_curand << <initBlocks, initThreads >> >(d_rng, RNG_SEED);
		CUDA_CALL(cudaDeviceSynchronize());
		init_agents << <initBlocks, initThreads >> >(d_rng, d_agents_init);
		//Free curand
		CUDA_CALL(cudaFree(d_rng));
	}
	//Init (appropriate SP constants)
	{
		CUDA_CALL(cudaMemcpyToSymbol(d_agentCount, &POP_SIZE, sizeof(unsigned int)));
		const float ONE = 1.0f;
		const float rSin45 = (float)(ONE*sin(glm::radians(45.0f)));
		CUDA_CALL(cudaMemcpyToSymbol(d_RADIUS, &ONE, sizeof(float)));
		CUDA_CALL(cudaMemcpyToSymbol(d_R_SIN_45, &rSin45, sizeof(float)));
	}	
	//Create & open log file
	std::ofstream logF;
	{
		std::string path = std::string(logName) + std::to_string(time(nullptr)) + ".csv";
		logF.open(path);
	}
	//Init log file (output run config to first line, output column headers to second line)
	{
		//Config
#ifdef _DEBUG
		logF << "{DEBUG COMPILATION}, ";
#endif
		logF << "[Static Mode], ";
		logF << "Population Size: " << POP_SIZE << ", ";
		logF << "Start Bins: " << INIT_BINS << ", ";
		logF << "End Bins: " << FINAL_BINS << ", ";
		logF << "Start Dims: (" << INIT_DIMS.x << ", " << INIT_DIMS.y << "), ";
		logF << "End Dims: (" << FINAL_DIMS.x << ", " << FINAL_DIMS.y << "), ";
		logF << "Average Neighbours: " << AVERAGE_NEIGHBOURS << "\n";
		//Config Header
		unsigned int i = 0;
		logF << "(" << (i++) << ") " << "Dim x,";
		logF << "(" << (i++) << ") " << "Dim y,";
		logF << "(" << (i++) << ") " << "Bins,";
		logF << "(" << (i++) << ") " << "Population Size,";
		//Default Header
		original::logHeader(logF, i);
		//Atomic Header
		atomic::logHeader(logF, i);
		logF << "\n";
	}
	//For-each step
	for(unsigned int t = 0;t<=STEPS;++t)
	{
		printf("\r%u/%u", t, STEPS);
		const glm::uvec2 t_DIMS = INIT_DIMS + glm::uvec2(round(STEP_DIMS.x*t),round(STEP_DIMS.y*t)); 
		const unsigned int t_BINS = glm::compMul(t_DIMS);
		//Init per step constants
		{
			CUDA_CALL(cudaMemcpyToSymbol(d_gridDim, &t_DIMS, sizeof(glm::uvec2)));
			glm::vec2 dims_float = glm::vec2(t_DIMS);
			CUDA_CALL(cudaMemcpyToSymbol(d_gridDim_float, &dims_float, sizeof(glm::vec2)));
		}
		//Log config
		{
			logF << t_DIMS.x << ",";
			logF << t_DIMS.y << ",";
			logF << t_BINS << ",";
			logF << POP_SIZE << ",";
		}
		original::Times defaultT;
		defaultT.overall = FLT_MAX;
		for (unsigned int i = 0; i<RETRIES; ++i)
		{
			//Reset actor pop (this *should* be redundant)
			CUDA_CALL(cudaMemcpy(d_agents_in, d_agents_init, sizeof(glm::vec2)*POP_SIZE, cudaMemcpyDeviceToDevice));
			//Run Default
			auto _defaultT = original::construct(POP_SIZE, t_DIMS, t_BINS);
			//Save quickest run
			defaultT = defaultT.overall < _defaultT.overall ? defaultT : _defaultT;
		}
		//Log Default
		original::logResult(logF, defaultT);

		atomic::Times atomicT;
		atomicT.overall = FLT_MAX;
		for (unsigned int i = 0; i<RETRIES; ++i)
		{
			//Reset actor pop (this *should* be redundant)
			CUDA_CALL(cudaMemcpy(d_agents_in, d_agents_init, sizeof(glm::vec2)*POP_SIZE, cudaMemcpyDeviceToDevice));
			//Run Atomic
			auto _atomicT = atomic::construct(POP_SIZE, t_DIMS, t_BINS);
			//Save quickest run
			atomicT = atomicT.overall < _atomicT.overall ? atomicT : _atomicT;
		}
		//Log Atomic
		atomic::logResult(logF, atomicT);
		//Newline log
		logF << "\n";
	}
	//Release memory
	{
		//Agents
		CUDA_CALL(cudaFree(d_agents_in));
		CUDA_CALL(cudaFree(d_agents_out));
		//Agent reset
		CUDA_CALL(cudaFree(d_agents_init));
		//SP interim
		CUDA_CALL(cudaFree(d_keys));
		CUDA_CALL(cudaFree(d_vals));
		//Final PBM storage
		CUDA_CALL(cudaFree(d_PBM_counts));
		CUDA_CALL(cudaFree(d_PBM));
		//original.cuh swap space
		CUDA_CALL(cudaFree(d_keys_swap));
		CUDA_CALL(cudaFree(d_vals_swap));
	}
	printf("\r%s Completed!\n", logName);
}
void runDynamic(const unsigned int &POP_SIZE, const unsigned int &START_BINS, const unsigned int &END_BINS, const unsigned int &STEPS, const char *logName)
{
	//Defined out of scope of methods being tested
	glm::vec2 *d_agents_init;
	curandState *d_rng = nullptr;
	//Pre-calc init final state
	const glm::uvec2 INIT_DIMS = glm::uvec2(static_cast<unsigned int>(floor(sqrt(START_BINS))));
	const unsigned int INIT_BINS = glm::compMul(INIT_DIMS);
	const glm::uvec2 FINAL_DIMS = glm::uvec2(INIT_DIMS.x, ceil(static_cast<float>(END_BINS) / INIT_DIMS.x));
	const unsigned int FINAL_BINS = glm::compMul(FINAL_DIMS);
	const glm::vec2 STEP_DIMS = glm::vec2(0, static_cast<float>(FINAL_DIMS.y - INIT_DIMS.y) / STEPS);
	const unsigned int AVERAGE_NEIGHBOURS = static_cast<unsigned int>(POP_SIZE * glm::pi<float>() / INIT_BINS);
	//Allocate SP memory
	{
		//Agents
		CUDA_CALL(cudaMalloc(&d_agents_in, sizeof(glm::vec2) * POP_SIZE));//Redundant, theoretically d_agents_in should remain unchanged.
		CUDA_CALL(cudaMalloc(&d_agents_out, sizeof(glm::vec2) * POP_SIZE));
		//Agent reset
		CUDA_CALL(cudaMalloc(&d_agents_init, sizeof(glm::vec2) * POP_SIZE));
		//SP interim
		CUDA_CALL(cudaMalloc(&d_keys, sizeof(unsigned int) * POP_SIZE));
		CUDA_CALL(cudaMalloc(&d_vals, sizeof(unsigned int) * POP_SIZE));
		//Final PBM storage
		CUDA_CALL(cudaMalloc(&d_PBM_counts, (FINAL_BINS + 1) * sizeof(unsigned int)));
		CUDA_CALL(cudaMalloc(&d_PBM, (FINAL_BINS + 1) * sizeof(unsigned int)));
		//original.cuh swap space
		CUDA_CALL(cudaMalloc(&d_keys_swap, sizeof(unsigned int) * POP_SIZE));
		CUDA_CALL(cudaMalloc(&d_vals_swap, sizeof(unsigned int) * POP_SIZE));
	}
	//Init cuRand
	{
		//cuRand
		CUDA_CALL(cudaMalloc(&d_rng, POP_SIZE * sizeof(curandState)));
		//Arbitrary thread block sizes (speed not too important during one off initialisation)
		unsigned int initThreads = 512;
		unsigned int initBlocks = (POP_SIZE / initThreads) + 1;
		init_curand << <initBlocks, initThreads >> >(d_rng, RNG_SEED);
		CUDA_CALL(cudaDeviceSynchronize());
	}
	//Init (appropriate SP constants)
	{
		CUDA_CALL(cudaMemcpyToSymbol(d_agentCount, &POP_SIZE, sizeof(unsigned int)));
		const float ONE = 1.0f;
		const float rSin45 = (float)(ONE*sin(glm::radians(45.0f)));
		CUDA_CALL(cudaMemcpyToSymbol(d_RADIUS, &ONE, sizeof(float)));
		CUDA_CALL(cudaMemcpyToSymbol(d_R_SIN_45, &rSin45, sizeof(float)));
	}
	//Create & open log file
	std::ofstream logF;
	{
		std::string path = std::string(logName) + std::to_string(time(nullptr)) + ".csv";
		logF.open(path);
	}
	//Init log file (output run config to first line, output column headers to second line)
	{
		//Config
#ifdef _DEBUG
		logF << "{DEBUG COMPILATION}, ";
#endif
		logF << "[Dynamic Mode], ";
		logF << "Population Size: " << POP_SIZE << ", ";
		logF << "Start Bins: " << INIT_BINS << ", ";
		logF << "End Bins: " << FINAL_BINS << ", ";
		logF << "Start Dims: (" << INIT_DIMS.x << ", " << INIT_DIMS.y << "), ";
		logF << "End Dims: (" << FINAL_DIMS.x << ", " << FINAL_DIMS.y << "), ";
		logF << "Average Neighbours: " << AVERAGE_NEIGHBOURS << "\n";
		//Config Header
		unsigned int i = 0;
		logF << "(" << (i++) << ") " << "Dim x,";
		logF << "(" << (i++) << ") " << "Dim y,";
		logF << "(" << (i++) << ") " << "Bins,";
		logF << "(" << (i++) << ") " << "Population Size,";
		//Default Header
		original::logHeader(logF, i);
		//Atomic Header
		atomic::logHeader(logF, i);
		logF << "\n";
	}
	//For-each step
	for (unsigned int t = 0; t <= STEPS; ++t)
	{
		printf("\r%u/%u", t, STEPS);
		const glm::uvec2 t_DIMS = INIT_DIMS + glm::uvec2(round(STEP_DIMS.x*t), round(STEP_DIMS.y*t));
		const unsigned int t_BINS = glm::compMul(t_DIMS);
		//Init per step constants
		{
			CUDA_CALL(cudaMemcpyToSymbol(d_gridDim, &t_DIMS, sizeof(glm::uvec2)));
			glm::vec2 dims_float = glm::vec2(t_DIMS);
			CUDA_CALL(cudaMemcpyToSymbol(d_gridDim_float, &dims_float, sizeof(glm::vec2)));
		}
		//Init agents (must occur after d_gridDim_float has been updated)
		{
			//Arbitrary thread block sizes (speed not too important during one off initialisation)
			unsigned int initThreads = 512;
			unsigned int initBlocks = (POP_SIZE / initThreads) + 1;
			init_agents << <initBlocks, initThreads >> >(d_rng, d_agents_init);
		}
		//Log config
		{
			logF << t_DIMS.x << ",";
			logF << t_DIMS.y << ",";
			logF << t_BINS << ",";
			logF << POP_SIZE << ",";
		}
		original::Times defaultT;
		defaultT.overall = FLT_MAX;
		for(unsigned int i=0;i<RETRIES;++i)
		{
			//Reset actor pop (this *should* be redundant)
			CUDA_CALL(cudaMemcpy(d_agents_in, d_agents_init, sizeof(glm::vec2)*POP_SIZE, cudaMemcpyDeviceToDevice));
			//Run Default
			auto _defaultT = original::construct(POP_SIZE, t_DIMS, t_BINS);
			//Save quickest run
			defaultT = defaultT.overall < _defaultT.overall ? defaultT : _defaultT;
		}
		//Log Default
		original::logResult(logF, defaultT);

		atomic::Times atomicT;
		atomicT.overall = FLT_MAX;
		for (unsigned int i=0; i<RETRIES; ++i)
		{
			//Reset actor pop (this *should* be redundant)
			CUDA_CALL(cudaMemcpy(d_agents_in, d_agents_init, sizeof(glm::vec2)*POP_SIZE, cudaMemcpyDeviceToDevice));
			//Run Atomic
			auto _atomicT = atomic::construct(POP_SIZE, t_DIMS, t_BINS);
			//Save quickest run
			atomicT = atomicT.overall < _atomicT.overall ? atomicT : _atomicT;
		}
		//Log Atomic
		atomic::logResult(logF, atomicT);
		//Newline log
		logF << "\n";
	}
	//Release memory
	{
		//Agents
		CUDA_CALL(cudaFree(d_agents_in));
		CUDA_CALL(cudaFree(d_agents_out));
		//Agent reset
		CUDA_CALL(cudaFree(d_agents_init));
		//SP interim
		CUDA_CALL(cudaFree(d_keys));
		CUDA_CALL(cudaFree(d_vals));
		//Final PBM storage
		CUDA_CALL(cudaFree(d_PBM_counts));
		CUDA_CALL(cudaFree(d_PBM));
		//original.cuh swap space
		CUDA_CALL(cudaFree(d_keys_swap));
		CUDA_CALL(cudaFree(d_vals_swap));
		//Free curand
		CUDA_CALL(cudaFree(d_rng));
	}
	printf("\r%s Completed!\n", logName);
}