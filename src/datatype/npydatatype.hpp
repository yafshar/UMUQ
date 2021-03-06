#ifndef UMUQ_NPYDATATYPE_H
#define UMUQ_NPYDATATYPE_H
#ifdef HAVE_PYTHON

#if HAVE_PYTHON == 1
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#include <Python.h>

// To avoid the compiler warning
#ifdef NPY_NO_DEPRECATED_API
#undef NPY_NO_DEPRECATED_API
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

namespace umuq
{

/*!
 * \brief Numpy data types variable template wrapper for the given C++ types
 *
 * \tparam DataType Data type
 *
 */
template <typename DataType>
constexpr NPY_TYPES NPIDatatype = NPY_NOTYPE; // variable template

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
constexpr NPY_TYPES NPIDatatype<bool> = NPY_BOOL;

template <>
constexpr NPY_TYPES NPIDatatype<char> = NPY_BYTE;

template <>
constexpr NPY_TYPES NPIDatatype<std::string> = NPY_STRING;

template <>
constexpr NPY_TYPES NPIDatatype<int8_t> = NPY_INT8;

template <>
constexpr NPY_TYPES NPIDatatype<uint8_t> = NPY_UINT8;

template <>
constexpr NPY_TYPES NPIDatatype<int16_t> = NPY_SHORT;

template <>
constexpr NPY_TYPES NPIDatatype<uint16_t> = NPY_USHORT;

template <>
constexpr NPY_TYPES NPIDatatype<int32_t> = NPY_INT;

template <>
constexpr NPY_TYPES NPIDatatype<uint32_t> = NPY_ULONG;

template <>
constexpr NPY_TYPES NPIDatatype<int64_t> = NPY_INT64;

template <>
constexpr NPY_TYPES NPIDatatype<uint64_t> = NPY_UINT64;

template <>
constexpr NPY_TYPES NPIDatatype<float> = NPY_FLOAT;

template <>
constexpr NPY_TYPES NPIDatatype<double> = NPY_DOUBLE;

template <>
constexpr NPY_TYPES NPIDatatype<long double> = NPY_LONGDOUBLE;

} // namespace umuq

#endif // HAVE_PYTHON
#endif // UMUQ_NPYDATATYPE
