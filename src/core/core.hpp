#ifndef UMHBM_CORE_H
#define UMHBM_CORE_H

//Include this file where all configuration variables are defined.
#include "../umhbm_config.h"

//Include this file where all our macros are defined.
#include "macros.hpp"

// Defines symbols for compile-time detection of which instructions are used.
// UMHBM_VECTORIZE_YY is defined if and only if the instruction set YY is used
#define UMHBM_VECTORIZE
#define UMHBM_VECTORIZE_SSE
#define UMHBM_VECTORIZE_SSE2

// Detect sse3/ssse3/sse4:
// gcc and icc defines __SSE3__, ...
#ifdef __SSE3__
#define UMHBM_VECTORIZE_SSE3
#endif
#ifdef __SSSE3__
#define UMHBM_VECTORIZE_SSSE3
#endif
#ifdef __SSE4_1__
#define UMHBM_VECTORIZE_SSE4_1
#endif
#ifdef __SSE4_2__
#define UMHBM_VECTORIZE_SSE4_2
#endif

// include files
extern "C" {
#ifdef UMHBM_VECTORIZE_SSE3
#include <pmmintrin.h>
#endif
#ifdef UMHBM_VECTORIZE_SSSE3
#include <tmmintrin.h>
#endif
#ifdef UMHBM_VECTORIZE_SSE4_1
#include <smmintrin.h>
#endif
#ifdef UMHBM_VECTORIZE_SSE4_2
#include <nmmintrin.h>
#endif
} // end extern "C"

#ifdef UMHBM_HAS_OPENMP
#include <omp.h>
#endif

#define _XOPEN_SOURCE 700
#define _BSD_SOURCE 1

#include <sys/stat.h> //stat

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstdio>  //fopen, fgets, sscanf, sprintf
#include <climits>
#include <cmath>
#include <cstring> //strlen, strstr, strtok

#include <functional>
#include <iosfwd>
#include <string>
#include <limits>
#include <algorithm>   // for min/max:
#include <type_traits> // for std::is_nothrow_move_assignable
#include <iostream>    // for outputting debug info
#include <fstream>
#include <sstream>
#include <ios>
#include <iomanip>
#include <system_error>
#include <memory>

/*!
 * \brief Namespace containing all symbols from the %UMHBM library. 
 */
namespace UMHBM
{
inline static const char *SimdInstructionSetsInUse(void)
{
#if defined(UMHBM_VECTORIZE_SSE4_2)
    return "SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2";
#elif defined(UMHBM_VECTORIZE_SSE4_1)
    return "SSE, SSE2, SSE3, SSSE3, SSE4.1";
#elif defined(UMHBM_VECTORIZE_SSSE3)
    return "SSE, SSE2, SSE3, SSSE3";
#elif defined(UMHBM_VECTORIZE_SSE3)
    return "SSE, SSE2, SSE3";
#elif defined(UMHBM_VECTORIZE_SSE2)
    return "SSE, SSE2";
#else
    return "None";
#endif
}
}

/*! 
 * This is the main module of UMHBM
 */
#include "meta.hpp"

#endif // UMHBM_CORE_H
