#ifndef UMUQ_EIGENLIB_H
#define UMUQ_EIGENLIB_H

#include "../data/eigendatatype.hpp"

/*!
 * \brief Eigen map type is a new type to map the existing C++ memory buffer to an Eigen Matrix object 
 * The Map operation maps the existing memory region into the Eigen’s data structures.
 * 
 * \tparam T         Data type or Eigen::Matrix type
 * \tparam _Options  A combination of either Eigen::RowMajor or Eigen::ColMajor, and of either 
 *                   Eigen::AutoAlign or Eigen::DontAlign. The former controls storage order, 
 *                   and defaults to column-major. The latter controls alignment, which is 
 *                   required for vectorization. It defaults to aligning matrices except for 
 *                   fixed sizes that aren't a multiple of the packet size. 
 * 
 * NOTE: Use of template is flexible enough that one can use directly the arithmatic data type and _Options 
 *       to be used as an Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> or one can directly
 *       pass only the Eigen::Matrix as template parameters
 * 
 * For example:
 * 
 * Simply mapping a contiguous C++ memory buffer as a column-major Eigen Matrix object:
 * \code 
 * double A[12];
 * for(int i = 0; i < 12; ++i) A[i] = (double)i;
 * EigenMapType<double, Eigen::ColMajor> B(A, 3, 4); 
 * std::cout << B << std::endl;
 * \endcode
 * 
 * Output: 
 * 
 * \f[
 * \begin{matrix}
 * 0 & 3 & 6 & ~9 \\ 
 * 1 & 4 & 7 & 10 \\ 
 * 2 & 5 & 8 & 11
 * \end{matrix}
 * \f]
 * 
 * \code 
 * using EMd = Eigen::Matrix<double, 3, 4>;
 * EigenMapType<EMd> C(A, 3, 4); 
 * std::cout << C << std::endl;
 * \endcode
 * 
 * Output: 
 * 
 * \f[
 * \begin{matrix}
 * 0 & 3 & 6 & ~9 \\ 
 * 1 & 4 & 7 & 10 \\ 
 * 2 & 5 & 8 & 11
 * \end{matrix}
 * \f]
 *
 */
template <class T, int _Options = Eigen::RowMajor>
using EMapType = Eigen::Map<typename std::conditional<std::is_arithmetic<T>::value, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options>, T>::type>;

/*!
 * \brief Eigen map type constant is a new read-only map type to map the existing C++ memory buffer to an Eigen Matrix object 
 * The Map operation maps the existing memory region into the Eigen’s data structures.
 * 
 * \tparam T         Data type or Eigen::Matrix type
 * \tparam _Options  A combination of either Eigen::RowMajor or Eigen::ColMajor, and of either 
 *                   Eigen::AutoAlign or Eigen::DontAlign. The former controls storage order, 
 *                   and defaults to column-major. The latter controls alignment, which is 
 *                   required for vectorization. It defaults to aligning matrices except for 
 *                   fixed sizes that aren't a multiple of the packet size. 
 * 
 * NOTE: Use of template is flexible enough that one can use directly the arithmatic data type and _Options 
 *       to be used as an Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> or one can directly
 *       pass only the Eigen::Matrix as template parameters
 * 
 */
template <typename T, int _Options = Eigen::RowMajor>
using EMapTypeConst = Eigen::Map<typename std::conditional<std::is_arithmetic<T>::value, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options>, T>::type const>;

/*!
 * \brief Eigen row vector map type is a new type, used to map the existing C++ memory buffer to an Eigen RowMajor Vector object
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam T Data type
 */
template <typename T>
using ERowVectorMapType = Eigen::Map<ERowVectorX<T>>;

/*!
 * \brief Eigen row vector constant map type is a new read-only map type to map the existing C++ memory buffer to an Eigen RowMajor Vector object
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam T Data type
 */
template <typename T>
using ERowVectorMapTypeConst = Eigen::Map<ERowVectorX<T> const>;

/*!
 * \brief Eigen vector map type is a new type to map the existing C++ memory buffer to an Eigen Vector object
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam T Data type
 */
template <typename T>
using EVectorMapType = Eigen::Map<EVectorX<T>>;

/*!
 * \brief New read-only map type is used to map the existing C++ memory buffer to an Eigen Vector object
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam T Data type
 */
template <typename T>
using EVectorMapTypeConst = Eigen::Map<EVectorX<T> const>;

/*!
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 *  
 * \tparam T        Eigen matrix type (dynamic_size_storage matrix)
 * 
 * \param  dataPtr  Pointer to the array of data
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * 
 * \returns  Eigen Matrix representation of the array    
 * 
 * 
 * NOTE:
 * If the T template class is a dynamic_size_storage Eigen::Matrix, then one should 
 * provide the number of rows and number of columns at input
 * 
 */
template <class T>
inline typename std::enable_if<T::MaxRowsAtCompileTime == Eigen::Dynamic || T::MaxColsAtCompileTime == Eigen::Dynamic, T>::type
EMap(typename T::Scalar *dataPtr, int const nRows, int const nCols)
{
	return EMapTypeConst<T>(dataPtr, nRows, nCols);
}

/*!
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 * 
 * \tparam T       Eigen matrix type (fixed_size_storage matrix)
 * 
 * \param dataPtr  Pointer to the array of data
 * \return T       Eigen Matrix representation of the array
 * 
 * NOTE:
 * If the T template class is a fixed_size_storage Eigen::Matrix, then one should not
 * provide the number of rows and number of columns at input
 */
template <class T>
inline T EMap(typename T::Scalar *dataPtr)
{
	return EMapTypeConst<T>(dataPtr);
}

/*!
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 * Eigen map function copies the existing C++ memory buffer to a temporary Eigen Matrix object 
 * of size(nRows, nCols). 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam T        Eigen matrix type (dynamic_size_storage matrix)
 * 
 * \param  dataPtr  Pointer to the array of data
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * 
 * \returns  Eigen Matrix representation of the array    
 *
 * NOTE:
 * If the T template class is a dynamic_size_storage Eigen::Matrix, then the size does must  
 * be passed to the constructor, because it is not specified by the Matrix type.
 */
template <class T>
inline typename std::enable_if<T::MaxRowsAtCompileTime == Eigen::Dynamic || T::MaxColsAtCompileTime == Eigen::Dynamic, T>::type
EMap(typename T::Scalar **dataPtr, int const nRows, int const nCols)
{
	//! We have a dynamic_size_storage matrix and it should get the size from number of rows and columns on input
	T tmpMatrix(nRows, nCols);
	for (int i = 0; i < nRows; i++)
	{
		tmpMatrix.row(i) = EVectorMapTypeConst<typename T::Scalar>(&dataPtr[i][0], nCols);
	}
	return tmpMatrix;
}

/*!
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 * Eigen map function copies the existing C++ memory buffer to a temporary Eigen Matrix object 
 * of size(nRows, nCols). 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 * 
 * \tparam T       Eigen matrix type (fixed_size_storage matrix)
 * 
 * \param dataPtr  Pointer to the array of data
 * 
 * \returns  Eigen Matrix representation of the array    
 *
 * NOTE:
 * If the T template class is a fixed_size_storage Eigen::Matrix, then the size does not have 
 * to be passed to the constructor, because it is already specified by the Matrix type.
 */
template <class T>
inline T EMap(typename T::Scalar **dataPtr)
{
	//! We have a fixed_size_storage matrix
	T tmpMatrix;
	auto nCols = tmpMatrix.cols();
	for (auto i = 0; i < tmpMatrix.rows(); i++)
	{
		tmpMatrix.row(i) = EVectorMapTypeConst<typename T::Scalar>(&dataPtr[i][0], nCols);
	}
	return tmpMatrix;
}


//! TODO:
//! We should add the arraywrapper with stride to not copy the data when it is not required

/*!
 * \brief Eigen map function copies the Eigen Matrix data to the array of data
 * Eigen map function copies the existing Eigen Matrix object to a C++ memory buffer of 
 * the same size as Eigen matrix.
 * 
 * \tparam T       Eigen matrix type
 * 
 * \param dataPtr  Pointer to the array of the same Eigen matrix element type with the same size 
 *                 The data from eMatrix are copied to dataPtr in a rowmajor
 * \param eMatrix  Eigen matrix
 * 
 * NOTE:
 * We have to copy the data as we do not know before hand that the internal Eigen matrix data 
 * pointer is Aligned, or Unaligned and what is the StrideType
 */
template <class T>
inline void EMap(typename T::Scalar *dataPtr, T const &eMatrix)
{
	EMapType<typename T::Scalar>(dataPtr, eMatrix.rows(), eMatrix.cols()) = eMatrix;
}

/*!
 * \brief Eigen map function copies the Eigen Matrix data to the array of data
 * Eigen map function copies the existing Eigen Matrix object to a C++ memory buffer of 
 * the same size as Eigen matrix.
 * 
 * \tparam T       Eigen matrix type
 * 
 * \param dataPtr  Pointer to the array of the same Eigen matrix element type with the same size 
 *                 The data from eMatrix are copied to dataPtr in a rowmajor
 * \param eMatrix  Eigen matrix
 * 
 * NOTE:
 * We have to copy the data as we do not know before hand that the internal Eigen matrix data 
 * pointer is Aligned, or Unaligned and what is the StrideType
 */
template <class T>
inline void EMap(typename T::Scalar **dataPtr, T const &eMatrix)
{
	for (auto i = 0; i < eMatrix.rows(); i++)
	{
		EVectorMapType<typename T::Scalar>(&dataPtr[i][0], eMatrix.cols()) = eMatrix.row(i);
	}
}

#endif // UMUQ_EIGENLIB_H
