#ifndef UMUQ_CORE_H
#define UMUQ_CORE_H

#ifdef HAVE_CONFIG_H
// Include this file where all configuration variables are defined.
#include <UMUQ_config.h>
#endif // HAVE_CONFIG

#ifdef HAVE_PYTHON
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif // _POSIX_C_SOURCE
#ifndef _AIX
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif // _XOPEN_SOURCE
#endif // _AIX

// Include Python.h before any standard headers are included
#ifdef PY_SSIZE_T_CLEAN
#undef PY_SSIZE_T_CLEAN
#endif // PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include <pythread.h>

// To avoid the compiler warning
#ifdef NPY_NO_DEPRECATED_API
#undef NPY_NO_DEPRECATED_API
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif // HAVE_PYTHON

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

// Include this file where all our macros are defined.
#include "macros.hpp"

// include files
extern "C"
{
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

#ifdef HAVE_TORC
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#ifdef _BSD_SOURCE
#undef _BSD_SOURCE
#endif

#define _XOPEN_SOURCE 700
#define _BSD_SOURCE 1
#include <torc.h>
#undef _XOPEN_SOURCE
#undef _BSD_SOURCE
#endif // HAVE_TORC

/*!
 * \brief Namespace containing all symbols from the %UMUQ library.
 */
namespace umuq
{

/*!
 * \defgroup Core_Module Core module
 * This is the core module of %UMUQ providing internal and core support
 */

/*!
 * \namespace umuq::internal
 * \ingroup Core_Module
 *
 * \brief Namespace containing all internal symbols from the %UMUQ library.
 *
 */
namespace internal
{

/*!
 * \ingroup Core_Module
 *
 * \brief SIMD instructions
 *
 * \returns const char*
 */
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

} // namespace internal
} // namespace umuq

/*!
 * \ingroup Core_Module
 *
 * \brief Default digits10
 */
#include "digits10.hpp"

#include "constants.hpp"

/*!
 * \ingroup Core_Module
 *
 * \brief This is the main meta module of UMUQ
 */
#include "meta.hpp"

/*!
 * \defgroup Test_Module Test module
 * This is the test module of %UMUQ providing functionality and classes for a unit testing.
 */

#endif // UMUQ_CORE
