#ifndef UMUQ_NPYDATATYPE_H
#define UMUQ_NPYDATATYPE_H
#ifdef HAVE_PYTHON

/*!
 * \brief numpy data types variable template wrapper for the given C++ types
 * 
 * \tparam DataType Data type
 * 
 */
template <typename DataType>
constexpr NPY_TYPES NPIDatatype = NPY_NOTYPE; // variable template

/*!
 * \brief Explicit instantiation for
 * 
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
 * - \b long double
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

#endif // HAVE_PYTHON
#endif // UMUQ_NPYDATATYPE
