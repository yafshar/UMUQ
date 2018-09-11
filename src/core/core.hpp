#ifndef UMUQ_CORE_H
#define UMUQ_CORE_H

#ifdef HAVE_CONFIG_H
//Include this file where all configuration variables are defined.
#include <UMUQ_config.h>
#endif /* HAVE_CONFIG_H */

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

#if HAVE_MPI == 1
#include <mpi.h>
#endif //MPI

//Include this file where all our macros are defined.
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
#include <iostream>	// for outputting debug info
#include <fstream>
#include <sstream>
#include <ios>
#include <iomanip>
#include <system_error>
#include <memory>
#include <random>
#include <map>
#include <mutex>

#if HAVE_PYTHON == 1
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#if PY_MAJOR_VERSION >= 3
#define PyString_FromString PyUnicode_FromString
#endif
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#ifdef _BSD_SOURCE
#undef _BSD_SOURCE
#endif
#define _XOPEN_SOURCE 700
#define _BSD_SOURCE 1
#ifdef HAVE_TORC
#include <torc.h>
#endif

/*!
 * \brief Namespace containing all symbols from the %UMUQ library. 
 */
namespace umuq
{
namespace internal
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

} // namespace internal

#ifdef M_PI
#undef M_PI
#endif
#ifdef M_2PI
#undef M_2PI
#endif
#ifdef M_SPI
#undef M_SPI
#endif
#ifdef M_S2PI
#undef M_S2PI
#endif
#ifdef M_LPI
#undef M_LPI
#endif
#ifdef M_L2PI
#undef M_L2PI
#endif
#ifdef LINESIZE
#undef LINESIZE
#endif

/*!
 * \brief Constant values of \f$ \pi, 2\pi, \sqrt{\pi}, \sqrt{2\pi}, ln(\pi), and ln(2\pi) \f$
 *
 * Reference:
 * http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html 
 */
//! \f$ \pi \f$
#define M_PI 3.14159265358979323846264338327950288419716939937510582097494459230781640l
//! \f$ 2\pi \f$
#define M_2PI 6.28318530717958647692528676655900576839433879875021164194988918461563281l
//! \f$ \sqrt{\pi} \f$
#define M_SPI 1.77245385090551602729816748334114518279754945612238712821380778985291128l
//! \f$ \sqrt{2\pi} \f$
#define M_S2PI 2.50662827463100050241576528481104525300698674060993831662992357634229365l
//! \f$ \log{\pi} \f$
#define M_LPI 1.14472988584940017414342735135305871164729481291531157151362307147213774l
//! \f$ \log{2\pi} \f$
#define M_L2PI 1.83787706640934548356065947281123527972279494727556682563430308096553139l

//! Maximum size of a char * in UMUQ parser & io
#define LINESIZE 256

} // namespace umuq

/*! 
 * This is the main module of UMUQ
 */
#include "meta.hpp"

/*!
 * Default digits10
 */
#include "digits10.hpp"

namespace umuq
{

/*!
 * \brief Get the machine precision accuracy for T data type
 * 
 * \tparam T Data type
 */
template <typename T>
static T machinePrecision = std::pow(T{10}, -digits10<T>());

} // namespace umuq

#endif // UMUQ_CORE_H
