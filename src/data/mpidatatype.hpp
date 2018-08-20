#ifndef UMUQ_MPIDATATYPE_H
#define UMUQ_MPIDATATYPE_H

/*! 
 * \brief MPI data types variable template wrapper for the given C++ type.
 *
 * \tparam T Data type
 * 
 */
template <typename T>
constexpr MPI_Datatype MPIDatatype = MPI_DATATYPE_NULL; // variable template

/*!
 * \brief Explicit instantiation for:
 * 
 * \b char
 * \b signed short
 * \b int
 * \b signed long
 * \b signed long long
 * \b float
 * \b double
 * \b long double
 * \b unsigned char
 * \b unsigned short
 * \b unsigned int
 * \b unsigned long
 * 
 * TODO: Complete the list
 * Any valid MPI data type value must have a corresponding explicit template instantiation below.
 */
template <>
constexpr MPI_Datatype MPIDatatype<char> = MPI_CHAR;

template <>
constexpr MPI_Datatype MPIDatatype<signed short> = MPI_SHORT;

template <>
constexpr MPI_Datatype MPIDatatype<int> = MPI_INT;

template <>
constexpr MPI_Datatype MPIDatatype<signed long> = MPI_LONG;

template <>
constexpr MPI_Datatype MPIDatatype<signed long long> = MPI_LONG_LONG_INT;

template <>
constexpr MPI_Datatype MPIDatatype<float> = MPI_FLOAT;

template <>
constexpr MPI_Datatype MPIDatatype<double> = MPI_DOUBLE;

template <>
constexpr MPI_Datatype MPIDatatype<long double> = MPI_LONG_DOUBLE;

template <>
constexpr MPI_Datatype MPIDatatype<unsigned char> = MPI_UNSIGNED_CHAR;

template <>
constexpr MPI_Datatype MPIDatatype<unsigned short> = MPI_UNSIGNED_SHORT;

template <>
constexpr MPI_Datatype MPIDatatype<unsigned int> = MPI_UNSIGNED;

template <>
constexpr MPI_Datatype MPIDatatype<unsigned long> = MPI_UNSIGNED_LONG;

#endif
