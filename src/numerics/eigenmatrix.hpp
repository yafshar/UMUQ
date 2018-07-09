#ifndef UMUQ_EIGENMATRIX_H
#define UMUQ_EIGENMATRIX_H

#include <Eigen/Dense>

/*!
 * \brief A convenience matrix data type 
 * An Eigen matrix type with dynamic sizes.
 * 
 * \tparam T  Data type 
 */
template <typename T>
using EMatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

/*!
 * \brief An Eigen matrix of doubles data type
 * 
 */
using EMatrixXd = EMatrixX<double>;

/*!
 * \brief A convenience matrix data type to cover the usual cases
 * 
 * \tparam T  Data type
 * 
 * \b EMatrixn : E + Matrix + n=(2, 3, 4, 5, or 6)
 * E is the abbreviation for Eigen followed by Matrix and any number of (2, 3, 4, 5, or 6) 
 * A rectangular matrix of T types of n*n=(2*2, 3*3, 4*4, 5*5, or 6*6) size.
 * 
 * For example:
 * EMatrix2<double> is a Eigen::Matrix of doubles with size of 2*2.
 * EMatrix5<int>    is a Eigen::Matrix of integers with size of 5*5.
 * 
 * \b EMatrixnX : E + Matrix + n=(2, 3, 4, 5, or 6) + X
 * E followed by Matrix and any number of n=(2, 3, 4, 5, or 6) and X
 * A rectangular matrix of type T with row size of n=(2, 3, 4, 5, or 6) and dynamic size columns
 * 
 * \b EMatrixXn : E + Matrix + X + n=(2, 3, 4, 5, or 6)
 * E followed by Matrix and X and any number of (2, 3, 4, 5, or 6)
 * A rectangular matrix of type T with dynamic size rows and column numbers of n=(2, 3, 4, 5, or 6)
 * 
 */
template <typename T>
using EMatrix2 = Eigen::Matrix<T, 2, 2>;
template <typename T>
using EMatrix2X = Eigen::Matrix<T, 2, Eigen::Dynamic>;
template <typename T>
using EMatrixX2 = Eigen::Matrix<T, Eigen::Dynamic, 2>;

template <typename T>
using EMatrix3 = Eigen::Matrix<T, 3, 3>;
template <typename T>
using EMatrix3X = Eigen::Matrix<T, 3, Eigen::Dynamic>;
template <typename T>
using EMatrixX3 = Eigen::Matrix<T, Eigen::Dynamic, 3>;

template <typename T>
using EMatrix4 = Eigen::Matrix<T, 4, 4>;
template <typename T>
using EMatrix4X = Eigen::Matrix<T, 4, Eigen::Dynamic>;
template <typename T>
using EMatrixX4 = Eigen::Matrix<T, Eigen::Dynamic, 4>;

template <typename T>
using EMatrix5 = Eigen::Matrix<T, 5, 5>;
template <typename T>
using EMatrix5X = Eigen::Matrix<T, 5, Eigen::Dynamic>;
template <typename T>
using EMatrixX5 = Eigen::Matrix<T, Eigen::Dynamic, 5>;

template <typename T>
using EMatrix6 = Eigen::Matrix<T, 6, 6>;
template <typename T>
using EMatrix6X = Eigen::Matrix<T, 6, Eigen::Dynamic>;
template <typename T>
using EMatrixX6 = Eigen::Matrix<T, Eigen::Dynamic, 6>;

/*!
 * \brief A convenience row-vector data type
 * An Eigen row-vector data type with dynamic size
 * 
 * \tparam T  Data type
 */
template <typename T>
using ERowVectorX = Eigen::Matrix<T, 1, Eigen::Dynamic>;

/*!
 * \brief An Eigen row-vector of doubles data type
 * 
 */
using ERowVectorXd = ERowVectorX<double>;

/*!
 * \brief A convenience row-vector data type to cover the usual cases
 * 
 * \tparam T  Data type
 * 
 * \b ERowVectorn : E + RowVector + n=(2, 3, 4, 5, or 6)
 * E followed by RowVector is a row-vector 
 * 
 * For example:
 * ERowVector6<float> is a row-vector of 6 floats.
 * 
 */
template <typename T>
using ERowVector2 = Eigen::Matrix<T, 1, 2>;
template <typename T>
using ERowVector3 = Eigen::Matrix<T, 1, 3>;
template <typename T>
using ERowVector4 = Eigen::Matrix<T, 1, 4>;
template <typename T>
using ERowVector5 = Eigen::Matrix<T, 1, 5>;
template <typename T>
using ERowVector6 = Eigen::Matrix<T, 1, 6>;

/*!
 * \brief A convenience column-vector data type 
 * An Eigen column-vector type with dynamic size.
 * 
 * \tparam T  Data type 
 */
template <typename T>
using EVectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

/*!
 * \brief An Eigen column-vector of doubles data type
 * 
 */
using EVectorXd = EVectorX<double>;

/*!
 * \brief A convenience column-vector data type to cover the usual cases
 * 
 * \tparam T Data type
 * 
 * 
 * \b EVectorn : E + Vector + n=(2, 3, 4, 5, or 6)
 * E followed by Vector is a column-vector
 * 
 * For example:
 * EVector3<int> is a column-vector of 3 integers.
 * 
 */
template <typename T>
using EVector2 = Eigen::Matrix<T, 2, 1>;
template <typename T>
using EVector3 = Eigen::Matrix<T, 3, 1>;
template <typename T>
using EVector4 = Eigen::Matrix<T, 4, 1>;
template <typename T>
using EVector5 = Eigen::Matrix<T, 5, 1>;
template <typename T>
using EVector6 = Eigen::Matrix<T, 6, 1>;

/*!
 * \brief Stores a set of parameters controlling the way matrices are printed
 * 
 * - precision \c FullPrecision.
 * - coeffSeparator string printed between two coefficients of the same row
 * - rowSeparator string printed between two rows
 */
Eigen::IOFormat fmt(Eigen::FullPrecision);

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
 * \brief Eigen map returns the Eigen Matrix representation of the array 
 *  
 * \tparam T        Eigen matrix type (dynamic, fixed size or any Eigen matrix type variation.)
 * 
 * \param  dataPtr  Pointer to the array of data
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * 
 * \returns  Eigen Matrix representation of the array    
 * 
 */
template <class T>
inline typename std::enable_if<T::MaxRowsAtCompileTime == Eigen::Dynamic || T::MaxColsAtCompileTime == Eigen::Dynamic, T>::type
EMap(typename T::Scalar *dataPtr, int const nRows, int const nCols)
{
	return EMapTypeConst<T>(dataPtr, nRows, nCols);
}

template <class T>
inline T EMap(typename T::Scalar *dataPtr)
{
	return EMapTypeConst<T>(dataPtr);
}

/*!
 * \brief Eigen map function copies the existing C++ memory buffer to an Eigen Matrix object of size(nRows, nCols) 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam T        Eigen matrix type (dynamic, fixed size or any Eigen matrix type variation.)
 * 
 * \param  dataPtr  Pointer to the array of data
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * 
 * \returns  Eigen Matrix representation of the array    
 * 
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

/*!
 * \brief Pointer will now point to a beginning of a memory buffer of the Eigen’s data structures
 * It copies the existing Eigen Matrix object of size(nRows, nCols) to the allocated C++ memory buffer of size(nRows*nCols)
 * The Map operation maps the existing Eigen’s data structure to the memory buffer
 * 
 * \tparam T        Eigen matrix type
 * 
 * \param  dataPtr  Pointer to the array of the same T element type with the size of (nRows*nCols)
 * \param  eMatrix  Input Eigen matrix of type T
 * 
 */
template <class T>
inline void EMap(typename T::Scalar *dataPtr, T const &eMatrix)
{
	EMapType<typename T::Scalar>(dataPtr, eMatrix.rows(), eMatrix.cols()) = eMatrix;
}

template <class T>
inline void EMap(typename T::Scalar **dataPtr, T const &eMatrix)
{
	for (auto i = 0; i < eMatrix.rows(); i++)
	{
		EVectorMapType<typename T::Scalar>(&dataPtr[i][0], eMatrix.cols()) = eMatrix.row(i);
	}
}

#endif
