#ifndef UMUQ_EIGENLIB_H
#define UMUQ_EIGENLIB_H

#include "data/eigendatatype.hpp"

namespace umuq
{

/*!
 * \brief Eigen map type is a new type to map the existing C++ memory buffer to an Eigen Matrix object 
 * The Map operation maps the existing memory region into the Eigen’s data structures.
 * 
 * \tparam T         Data type or Eigen::Matrix type
 * 
 * The _Options template parameter is optional
 * 
 * \tparam _Options  A combination of either 
 *                   \b #Eigen::RowMajor or 
 *                   \b #Eigen::ColMajor, 
 *                     and of either
 *                   \b #Eigen::AutoAlign or 
 *                   \b #Eigen::DontAlign.
 *                   The former controls storage order, and defaults to column-major. The latter controls alignment, which is required
 *                   for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
 *
 * 
 * NOTE: Use of template is flexible enough that one can use directly the arithmetic data type and _Options 
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
 * 
 * The _Options template parameter is optional
 * 
 * \tparam _Options  A combination of either 
 *                   \b #Eigen::RowMajor or 
 *                   \b #Eigen::ColMajor, 
 *                     and of either
 *                   \b #Eigen::AutoAlign or 
 *                   \b #Eigen::DontAlign.
 *                   The former controls storage order, and defaults to column-major. The latter controls alignment, which is required
 *                   for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
 * 
 * NOTE: Use of template is flexible enough that one can use directly the arithmetic data type and _Options 
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
 * \tparam EigenMatrixT  Eigen matrix type (dynamic_size_storage matrix)
 * 
 * \param  dataPtr  Pointer to the array of data
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * 
 * \returns  Eigen Matrix representation of the array    
 * 
 * 
 * NOTE:
 * If the EigenMatrixT template class is a dynamic_size_storage Eigen::Matrix, then one should 
 * provide the number of rows and number of columns at input
 * 
 */
template <class EigenMatrixT>
inline typename std::enable_if<EigenMatrixT::MaxRowsAtCompileTime == Eigen::Dynamic || EigenMatrixT::MaxColsAtCompileTime == Eigen::Dynamic, EigenMatrixT>::type
EMap(typename EigenMatrixT::Scalar *dataPtr, int const nRows, int const nCols)
{
	return EMapTypeConst<EigenMatrixT>(dataPtr, nRows, nCols);
}

/*!
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 * 
 * \tparam EigenMatrixT Eigen matrix type (fixed_size_storage matrix)
 * 
 * \param dataPtr  Pointer to the array of data
 * \return EigenMatrixT  Eigen Matrix representation of the array
 * 
 * NOTE:
 * If the EigenMatrixT template class is a fixed_size_storage Eigen::Matrix, then one should not
 * provide the number of rows and number of columns at input
 */
template <class EigenMatrixT>
inline EigenMatrixT EMap(typename EigenMatrixT::Scalar *dataPtr)
{
	return EMapTypeConst<EigenMatrixT>(dataPtr);
}

/*!
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 * Eigen map function copies the existing C++ memory buffer to a temporary Eigen Matrix object 
 * of size(nRows, nCols). 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam EigenMatrixT Eigen matrix type (dynamic_size_storage matrix)
 * 
 * \param  dataPtr  Pointer to the array of data
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * 
 * \returns Eigen Matrix representation of the array    
 *
 * NOTE:
 * If the EigenMatrixT template class is a dynamic_size_storage Eigen::Matrix, then the size does must  
 * be passed to the constructor, because it is not specified by the Matrix type.
 */
template <class EigenMatrixT>
inline typename std::enable_if<EigenMatrixT::MaxRowsAtCompileTime == Eigen::Dynamic || EigenMatrixT::MaxColsAtCompileTime == Eigen::Dynamic, EigenMatrixT>::type
EMap(typename EigenMatrixT::Scalar **dataPtr, int const nRows, int const nCols)
{
	//! We have a dynamic_size_storage matrix and it should get the size from number of rows and columns on input
	EigenMatrixT tmpMatrix(nRows, nCols);
	for (int i = 0; i < nRows; i++)
	{
		tmpMatrix.row(i) = EVectorMapTypeConst<typename EigenMatrixT::Scalar>(&dataPtr[i][0], nCols);
	}
	return tmpMatrix;
}

/*!
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 * Eigen map function copies the existing C++ memory buffer to a temporary Eigen Matrix object 
 * of size(nRows, nCols). 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 * 
 * \tparam EigenMatrixT       Eigen matrix type (fixed_size_storage matrix)
 * 
 * \param dataPtr  Pointer to the array of data
 * 
 * \returns  Eigen Matrix representation of the array    
 *
 * NOTE:
 * If the EigenMatrixT template class is a fixed_size_storage Eigen::Matrix, then the size does not have 
 * to be passed to the constructor, because it is already specified by the Matrix type.
 */
template <class EigenMatrixT>
inline EigenMatrixT EMap(typename EigenMatrixT::Scalar **dataPtr)
{
	//! We have a fixed_size_storage matrix
	EigenMatrixT tmpMatrix;
	auto nCols = tmpMatrix.cols();
	for (auto i = 0; i < tmpMatrix.rows(); i++)
	{
		tmpMatrix.row(i) = EVectorMapTypeConst<typename EigenMatrixT::Scalar>(&dataPtr[i][0], nCols);
	}
	return tmpMatrix;
}

//! TODO:
//! We should add the arraywrapper with inner and outer stride to not copy the data when it is not required

/*!
 * \brief Eigen map function copies the Eigen Matrix data to the array of data
 * Eigen map function copies the existing Eigen Matrix object to a C++ memory buffer of 
 * the same size as Eigen matrix.
 * 
 * \tparam EigenMatrixT Eigen matrix type
 * 
 * \param dataPtr  Pointer to the array of the same Eigen matrix element type with the same size 
 *                 The data from eMatrix are copied to dataPtr in a rowmajor
 * \param eMatrix  Eigen matrix
 * 
 * NOTE:
 * We have to copy the data as we do not know before hand that the internal Eigen matrix data 
 * pointer is Aligned, or Unaligned and what is the StrideType
 */
template <class EigenMatrixT>
inline void EMap(typename EigenMatrixT::Scalar *dataPtr, EigenMatrixT const &eMatrix)
{
	EMapType<typename EigenMatrixT::Scalar>(dataPtr, eMatrix.rows(), eMatrix.cols()) = eMatrix;
}

/*!
 * \brief Eigen map function copies the Eigen Matrix data to the array of data
 * Eigen map function copies the existing Eigen Matrix object to a C++ memory buffer of 
 * the same size as Eigen matrix.
 * 
 * \tparam EigenMatrixT Eigen matrix type
 * 
 * \param dataPtr  Pointer to the array of the same Eigen matrix element type with the same size 
 *                 The data from eMatrix are copied to dataPtr in a rowmajor
 * \param eMatrix  Eigen matrix
 * 
 * NOTE:
 * We have to copy the data as we do not know before hand that the internal Eigen matrix data 
 * pointer is Aligned, or Unaligned and what is the StrideType
 */
template <class EigenMatrixT>
inline void EMap(typename EigenMatrixT::Scalar **dataPtr, EigenMatrixT const &eMatrix)
{
	for (auto i = 0; i < eMatrix.rows(); i++)
	{
		EVectorMapType<typename EigenMatrixT::Scalar>(&dataPtr[i][0], eMatrix.cols()) = eMatrix.row(i);
	}
}

/*!
 * \brief This is a check to see if a selfadjoint matrix is positive definite or not?
 * 
 * A matrix is selfadjoint if it equals its adjoint. For real matrices, this means 
 * that the matrix is symmetric: it equals its transpose.
 * 
 * \tparam T Data type
 * 
 * \param dataPtr Pointer to the array of data (nDim * nDim)
 * \param nDim    Dimension of a square matrix  
 * 
 * \return true  If the selfadjoint matrix is positive definite
 * \return false If the selfadjoint matrix is not positive definite
 */
template <typename T>
inline bool isSelfAdjointMatrixPositiveDefinite(T *dataPtr, int const nDim)
{
	//! First Map the data to Eigen Matrix format
	EMapType<T, Eigen::ColMajor> DMap(dataPtr, nDim, nDim);
	//! Since the matrix is selfadjoint
	Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(DMap, Eigen::EigenvaluesOnly);

	return es.eigenvalues()(0) > 0 && es.eigenvalues()(0) / es.eigenvalues()(nDim - 1) > machinePrecision<T>;
}

/*!
 * \brief This is a check to see if a selfadjoint matrix is positive definite or not?
 * 
 * \tparam EigenMatrixT Eigen Matrix type
 * 
 * \param eMatrix Input matrix
 * 
 * \return true  If the selfadjoint matrix is positive definite
 * \return false If the selfadjoint matrix is not positive definite
 */
template <class EigenMatrixT>
inline bool isSelfAdjointMatrixPositiveDefinite(EigenMatrixT const &eMatrix)
{
	//! Since the matrix is selfadjoint
	Eigen::SelfAdjointEigenSolver<EigenMatrixT> es(eMatrix, Eigen::EigenvaluesOnly);

	return es.eigenvalues()(0) > 0 && es.eigenvalues()(0) / es.eigenvalues()(eMatrix.rows() - 1) > machinePrecision<typename EigenMatrixT::Scalar>;
}

/*!
 * \brief Force the selfadjoint matrix to be positive definite   
 * 
 * \tparam T Data type
 * 
 * \param dataPtr Pointer to the array of data (nDim * nDim)
 * \param nDim    Dimension of a square matrix 
 */
template <typename T>
void forceSelfAdjointMatrixPositiveDefinite(T *dataPtr, int const nDim)
{
	//! First Map the data to Eigen Matrix format
	EMapType<T, Eigen::ColMajor> eMatrix(dataPtr, nDim, nDim);

	//! Fixed value to increase
	T const increaseRate(0.01);

	//! Fixed value to multiply by
	T const fixedRate(1.01);

	//! Vector for diagonal elements
	std::vector<T> D(nDim);

#ifdef DEBUG
	std::cout << "eMatrix=" << eMatrix << std::endl;
	std::size_t iter(0);
#endif

	bool belowMachinePrecision(false);
	for (int i = 0; i < nDim; i++)
	{
		D[i] = eMatrix(i, i);
		if (D[i] <= machinePrecision<T>)
		{
			//! We have negative or zero ( < machinePrecision) on the diagonal elements fo the matrix
			belowMachinePrecision = true;
		}
	}

	if (belowMachinePrecision)
	{
		//! Find the maximum absolute element on the diagonal of the matrix
		T MaxAbsD = *std::max_element(D.begin(), D.end(), [](T const &a, T const &b) { return (std::abs(a) < std::abs(b)); });

		//! Find the minimum element
		T const MinD = *std::min_element(D.begin(), D.end());

		MaxAbsD *= increaseRate;

		//! There is a negative component on the diagonal, and we should get rid of it.
		if (MinD < 0)
		{
			MaxAbsD -= MinD;
		}

		if (MaxAbsD < machinePrecision<T>)
		{
			MaxAbsD = machinePrecision<T>;
		}

		for (int i = 0; i < nDim; i++)
		{
			eMatrix(i, i) += MaxAbsD;
		}
		//! Now none of the diagonal elements are below machine precision
	}
	else
	{
		for (int i = 0; i < nDim; i++)
		{
			eMatrix(i, i) *= fixedRate;
		}
	}

	while (!isSelfAdjointMatrixPositiveDefinite<T>(dataPtr, nDim))
	{
#ifdef DEBUG
		iter++;
		std::cout << "Iteration number " << iter << " to force the Covariance Matrix Positive Definite" << std::endl;
		std::cout << "eMatrix=" << eMatrix << std::endl;
#endif
		for (int i = 0; i < nDim; i++)
		{
			eMatrix(i, i) *= fixedRate;
		}
	}
	return;
}

/*!
 * \brief Force the selfadjoint matrix to be positive definite   
 * 
 * \tparam EigenMatrixT Eigen Matrix type
 * 
 * \param eMatrix Input matrix
 */
template <class EigenMatrixT>
void forceSelfAdjointMatrixPositiveDefinite(EigenMatrixT &eMatrix)
{
	typedef typename EigenMatrixT::Scalar T;

	//! Fixed value to increase
	T const increaseRate(0.01);

	//! Fixed value to multiply by
	T const fixedRate(1.01);

	//! Size of the matrix
	Eigen::Index const nDim = eMatrix.rows();

	//! Vector for diagonal elements
	std::vector<T> D(nDim);

#ifdef DEBUG
	std::cout << "eMatrix=" << eMatrix << std::endl;
	std::size_t iter(0);
#endif

	bool belowMachinePrecision(false);
	for (Eigen::Index i = 0; i < nDim; i++)
	{
		D[i] = eMatrix(i, i);
		if (D[i] <= machinePrecision<T>)
		{
			//! We have negative or zero ( < machinePrecision) on the diagonal elements fo the matrix
			belowMachinePrecision = true;
		}
	}

	if (belowMachinePrecision)
	{
		//! Find the maximum absolute element on the diagonal of the matrix
		T MaxAbsD = *std::max_element(D.begin(), D.end(), [](T const &a, T const &b) { return (std::abs(a) < std::abs(b)); });

		//! Find the minimum element
		T const MinD = *std::min_element(D.begin(), D.end());

		MaxAbsD *= increaseRate;

		//! There is a negative component on the diagonal, and we should get rid of it.
		if (MinD < 0)
		{
			MaxAbsD -= MinD;
		}

		if (MaxAbsD < machinePrecision<T>)
		{
			MaxAbsD = machinePrecision<T>;
		}

		for (Eigen::Index i = 0; i < nDim; i++)
		{
			eMatrix(i, i) += MaxAbsD;
		}
		//! Now none of the diagonal elements are below machine precision
	}
	else
	{
		for (Eigen::Index i = 0; i < nDim; i++)
		{
			eMatrix(i, i) *= fixedRate;
		}
	}

	while (!isSelfAdjointMatrixPositiveDefinite<EigenMatrixT>(eMatrix))
	{
#ifdef DEBUG
		iter++;
		std::cout << "Iteration number " << iter << " to force the Covariance Matrix Positive Definite" << std::endl;
		std::cout << "eMatrix=" << eMatrix << std::endl;
#endif
		for (Eigen::Index i = 0; i < nDim; i++)
		{
			eMatrix(i, i) *= fixedRate;
		}
	}

	return;
}

} // namespace umuq

#endif // UMUQ_EIGENLIB
