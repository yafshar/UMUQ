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
#include <cstdio> //fopen, fgets, sscanf, sprintf
#include <climits>
#include <cmath>
#include <cstring> //strlen, strstr, strtok

#include <typeinfo>
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
#include <random>

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

#ifdef M_PI
#undef M_PI
#endif
// source: http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944592307816406l
#define M_2PI 6.283185307179586476925286766559005768394338798750211641949889184615632812l
#define M_LPI std::log(M_PI)
#define M_L2PI std::log(M_2PI)

/*! 
 * This is the main module of UMHBM
 */
#include "meta.hpp"

/*! 
 * Handles runtime error
 */
class UMUQexception : public std::runtime_error
{
  public:
    UMUQexception(const char *message) : std::runtime_error(message) {}

    UMUQexception(const std::string &message) : std::runtime_error(message) {}
};

#endif // UMHBM_CORE_H
