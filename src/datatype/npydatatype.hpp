#ifndef UMUQ_NPYDATATYPE_H
#define UMUQ_NPYDATATYPE_H
#ifdef HAVE_PYTHON

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

// Include Python.h before any standard headers are included
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// To avoid the compiler warning
#ifdef NPY_NO_DEPRECATED_API
#undef NPY_NO_DEPRECATED_API
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace umuq
{

/*!
 * \brief Numpy data types variable template wrapper for the given C++ types
 *
 * \tparam DataType Data type
 *
 */
template <typename DataType>
constexpr NPY_TYPES NPYDatatype = NPY_NOTYPE; // variable template

/*!
 * Explicit instantiation for the given C++ types of <br>
 * - \b bool
 * - \b char
 * - \b std::string
 * - \b int8_t
 * - \b uint8_t
 * - \b int16_t
 * - \b uint16_t
 * - \b int32_t
 * - \b uint32_t
 * - \b int64_t
 * - \b uint64_t
 * - \b float
 * - \b double
 * - \b long double <br>
 * to Numpy data types.
 *
 * \todo
 * Complete the list.
 * Any valid data type value must have a corresponding explicit template instantiation below.
 *
 */
template <>
constexpr NPY_TYPES NPYDatatype<bool> = NPY_BOOL;

template <>
constexpr NPY_TYPES NPYDatatype<char> = NPY_BYTE;

template <>
constexpr NPY_TYPES NPYDatatype<std::string> = NPY_STRING;

template <>
constexpr NPY_TYPES NPYDatatype<int8_t> = NPY_INT8;

template <>
constexpr NPY_TYPES NPYDatatype<uint8_t> = NPY_UINT8;

template <>
constexpr NPY_TYPES NPYDatatype<int16_t> = NPY_INT16;

template <>
constexpr NPY_TYPES NPYDatatype<uint16_t> = NPY_UINT16;

template <>
constexpr NPY_TYPES NPYDatatype<int32_t> = NPY_INT32;

template <>
constexpr NPY_TYPES NPYDatatype<uint32_t> = NPY_UINT32;

template <>
constexpr NPY_TYPES NPYDatatype<int64_t> = NPY_INT64;

template <>
constexpr NPY_TYPES NPYDatatype<uint64_t> = NPY_UINT64;

template <>
constexpr NPY_TYPES NPYDatatype<float> = NPY_FLOAT32;

template <>
constexpr NPY_TYPES NPYDatatype<double> = NPY_FLOAT64;

template <>
constexpr NPY_TYPES NPYDatatype<long double> = NPY_LONGDOUBLE;

} // namespace umuq

#endif // HAVE_PYTHON
#endif // UMUQ_NPYDATATYPE
