#ifndef UMUQ_CORE_H
#define UMUQ_CORE_H

#ifdef HAVE_CONFIG_H
//Include this file where all configuration variables are defined.
#include <UMUQ_config.h>
#endif /* HAVE_CONFIG_H */

//Include this file where all our macros are defined.
#include "macros.hpp"

// Defines symbols for compile-time detection of which instructions are used.
// UMUQ_VECTORIZE_YY is defined if and only if the instruction set YY is used
#define UMUQ_VECTORIZE
#define UMUQ_VECTORIZE_SSE
#define UMUQ_VECTORIZE_SSE2

// Detect sse3/ssse3/sse4:
// gcc and icc defines __SSE3__, ...
#ifdef __SSE3__
#define UMUQ_VECTORIZE_SSE3
#endif
#ifdef __SSSE3__
#define UMUQ_VECTORIZE_SSSE3
#endif
#ifdef __SSE4_1__
#define UMUQ_VECTORIZE_SSE4_1
#endif
#ifdef __SSE4_2__
#define UMUQ_VECTORIZE_SSE4_2
#endif

// include files
extern "C" {
#ifdef UMUQ_VECTORIZE_SSE3
#include <pmmintrin.h>
#endif
#ifdef UMUQ_VECTORIZE_SSSE3
#include <tmmintrin.h>
#endif
#ifdef UMUQ_VECTORIZE_SSE4_1
#include <smmintrin.h>
#endif
#ifdef UMUQ_VECTORIZE_SSE4_2
#include <nmmintrin.h>
#endif
} // end extern "C"

#ifdef UMUQ_HAS_OPENMP
#include <omp.h>
#endif

#define _XOPEN_SOURCE 700
#define _BSD_SOURCE 1

#ifdef HAVE_TORC
#include <torc.h>
#endif

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
#include <map>
#include <mutex>
#include <functional>

// #define HAVE_PYTHON 0
#ifdef HAVE_PYTHON
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#if PY_MAJOR_VERSION >= 3
#define PyString_FromString PyUnicode_FromString
#endif
#endif

/*!
 * \brief Namespace containing all symbols from the %UMUQ library. 
 */
namespace UMUQ
{

inline static const char *SimdInstructionSetsInUse(void)
{
#if defined(UMUQ_VECTORIZE_SSE4_2)
    return "SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2";
#elif defined(UMUQ_VECTORIZE_SSE4_1)
    return "SSE, SSE2, SSE3, SSSE3, SSE4.1";
#elif defined(UMUQ_VECTORIZE_SSSE3)
    return "SSE, SSE2, SSE3, SSSE3";
#elif defined(UMUQ_VECTORIZE_SSE3)
    return "SSE, SSE2, SSE3";
#elif defined(UMUQ_VECTORIZE_SSE2)
    return "SSE, SSE2";
#else
    return "None";
#endif
}

#ifdef M_PI
#undef M_PI
#endif

/*!
 * Reference:
 * http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html 
 */
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944592307816406l
#define M_2PI 6.283185307179586476925286766559005768394338798750211641949889184615632812l
#define M_SPI 1.772453850905516027298167483341145182797549456122387128213807789852911284l
#define M_S2PI 2.506628274631000502415765284811045253006986740609938316629923576342293654l
#define M_LPI std::log(M_PI)
#define M_L2PI std::log(M_2PI)

#define LINESIZE 256
} // namespace UMUQ

/*! 
 * This is the main module of UMUQ
 */
#include "meta.hpp"

#endif // UMUQ_CORE_H
