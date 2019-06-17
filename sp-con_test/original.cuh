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
		unsigned long long atomicSort = 0;
		unsigned long long sweep = 0;
		unsigned long long reorder = 0;
		unsigned long long tex = 0;
	};

	Times construct(const unsigned int &POPULATION_SIZE, const glm::uvec2 &DIMS, const unsigned int &BIN_COUNT)
	{
		Times t = {};

		return t;
	}
	void logHeader(std::ofstream &f, unsigned int &i)
	{
		f << "(" << (i++) << ") " << "Original_Overall,";
		//...
	}
	void logResult(std::ofstream &f, const Times &t)
	{
		f << t.overall << ",";
		f << t.histogram << ",";
		f << t.scan << ",";
		f << t.reorder << ",";
		f << t.tex << ",";
	}
}
#endif //__original_cuh__