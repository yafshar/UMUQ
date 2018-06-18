#ifndef UMUQ_MPIDATATYPE_H
#define UMUQ_MPIDATATYPE_H

/*! \class MPITypeWrapper
 * \brief This struct wrappers the MPI data type value for the given C++ type.
 *
 * Any valid MPI data type value must have a corresponding explicit template instantiation below.
 */
template <typename T>
struct MPITypeWrapper
{
	/*!
     * \brief Construct a new MPITypeWrapper object
     * 
     */
	MPITypeWrapper()
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " This type is not currently supported!" << std::endl;
		throw(std::runtime_error("Wrong type!"));
	}
	//! MPI data type
	MPI_Datatype type;
};

/*!
 * \brief Explicit instantiation for 
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
 * \b std::size_t
 */
template <>
MPITypeWrapper<char>::MPITypeWrapper() : type(MPI_CHAR) {}
template <>
MPITypeWrapper<signed short>::MPITypeWrapper() : type(MPI_SHORT) {}
template <>
MPITypeWrapper<int>::MPITypeWrapper() : type(MPI_INT) {}
template <>
MPITypeWrapper<signed long>::MPITypeWrapper() : type(MPI_LONG) {}
template <>
MPITypeWrapper<signed long long>::MPITypeWrapper() : type(MPI_LONG_LONG_INT) {}
template <>
MPITypeWrapper<float>::MPITypeWrapper() : type(MPI_FLOAT) {}
template <>
MPITypeWrapper<double>::MPITypeWrapper() : type(MPI_DOUBLE) {}
template <>
MPITypeWrapper<long double>::MPITypeWrapper() : type(MPI_LONG_DOUBLE) {}
template <>
MPITypeWrapper<unsigned char>::MPITypeWrapper() : type(MPI_UNSIGNED_CHAR) {}
template <>
MPITypeWrapper<unsigned short>::MPITypeWrapper() : type(MPI_UNSIGNED_SHORT) {}
template <>
MPITypeWrapper<unsigned int>::MPITypeWrapper() : type(MPI_UNSIGNED) {}
template <>
MPITypeWrapper<unsigned long>::MPITypeWrapper() : type(MPI_UNSIGNED_LONG) {}

/*!
 * \brief MPITypeof maps the type of T to an MPI data type, e.g. (float -> MPI_FLOAT), (double -> MPI_DOUBLE), etc..
 * 
 * \returns MPI_Datatype 
 */
inline MPI_Datatype MPITypeof(char) { return MPI_CHAR; }
inline MPI_Datatype MPITypeof(signed short) { return MPI_SHORT; }
inline MPI_Datatype MPITypeof(signed int) { return MPI_INT; }
inline MPI_Datatype MPITypeof(signed long) { return MPI_LONG; }
inline MPI_Datatype MPITypeof(signed long long) { return MPI_LONG_LONG_INT; }
inline MPI_Datatype MPITypeof(float) { return MPI_FLOAT; }
inline MPI_Datatype MPITypeof(double) { return MPI_DOUBLE; }
inline MPI_Datatype MPITypeof(long double) { return MPI_LONG_DOUBLE; }
inline MPI_Datatype MPITypeof(unsigned char) { return MPI_UNSIGNED_CHAR; }
inline MPI_Datatype MPITypeof(unsigned short) { return MPI_UNSIGNED_SHORT; }
inline MPI_Datatype MPITypeof(unsigned) { return MPI_UNSIGNED; }
inline MPI_Datatype MPITypeof(unsigned long) { return MPI_UNSIGNED_LONG; }

#endif
