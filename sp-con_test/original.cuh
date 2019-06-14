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

	Times construct()
	{
		Times t = {};

		return t;
	}
}
#endif //__original_cuh__