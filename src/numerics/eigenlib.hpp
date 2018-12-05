#ifndef UMUQ_EIGENLIB_H
#define UMUQ_EIGENLIB_H

#include "datatype/eigendatatype.hpp"

namespace umuq
{

inline namespace linearalgebra
{

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen map type \c #EMapType is a new type to map the existing C++ memory buffer to an Eigen Matrix object.
 * The Map operation maps the existing memory region into the Eigen’s data structures.
 * 
 * \tparam T        Data type or \b Eigen::Matrix type
 * \tparam Options  optional parameter, a combination of either 
 *                  - \b Eigen::RowMajor or \b Eigen::ColMajor, <br> 
 *                    or one of either <br>
 *                  - \b Eigen::AutoAlign, or \b Eigen::DontAlign. <br>
 *                  The former controls storage order, and defaults to column-major. The latter controls alignment, which is required
 *                  for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
 *
 * 
 * \note 
 * 	- Use of template is flexible enough that one can use directly the arithmetic data type and Options 
 *    to be used as an \c Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Options> or one can directly
 *    pass only the \c Eigen::Matrix as template parameters
 * 
 * Example: <br>
 * Simply mapping a contiguous C++ memory buffer as a column-major Eigen Matrix object:
 * \code 
 * double A[12];
 * for(int i = 0; i < 12; ++i) A[i] = (double)i;
 * 
 * EMapType<double, Eigen::ColMajor> B(A, 3, 4); 
 * 
 * std::cout << B << std::endl;
 * \endcode
 * 
 * Output: <br> 
 * \f$
 * \begin{matrix}
 * 0 & 3 & 6 & ~9 \\ 
 * 1 & 4 & 7 & 10 \\ 
 * 2 & 5 & 8 & 11
 * \end{matrix}
 * \f$
 * 
 * \code 
 * double A[12];
 * for(int i = 0; i < 12; ++i) A[i] = (double)i;
 * 
 * EMapType<double> B(A, 3, 4); 
 * 
 * std::cout << B << std::endl;
 * \endcode
 * 
 * Output: <br>
 * \f$
 * \begin{matrix}
 * 0 & 1 &  2 &  3 \\ 
 * 4 & 5 &  6 &  7 \\ 
 * 8 & 9 & 10 & 11
 * \end{matrix}
 * \f$
 * 
 * \code 
 * using EMd = Eigen::Matrix<double, 3, 4>;
 * EMapType<EMd> C(A, 3, 4); 
 * 
 * std::cout << C << std::endl;
 * \endcode
 * 
 * Output:<br>
 * \f$
 * \begin{matrix}
 * 0 & 3 & 6 & ~9 \\ 
 * 1 & 4 & 7 & 10 \\ 
 * 2 & 5 & 8 & 11
 * \end{matrix}
 * \f$
 */
template <class T, int Options = Eigen::RowMajor>
using EMapType = Eigen::Map<typename std::conditional<std::is_arithmetic<T>::value, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Options>, T>::type>;

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen map type constant is a new read-only map type to map the existing C++ memory buffer to an Eigen Matrix object 
 * The Map operation maps the existing memory region into the Eigen’s data structures.
 * 
 * \tparam T        Data type or \b Eigen::Matrix type
 * \tparam Options  optional parameter, a combination of either 
 *                  - \b Eigen::RowMajor or \b Eigen::ColMajor, <br>
 *                    or one of either <br>
 *                  - \b Eigen::AutoAlign or \b Eigen::DontAlign. <br>
 *                  The former controls storage order, and defaults to column-major. The latter controls alignment, which is required
 *                  for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
 * 
 * \note 
 * - Use of template is flexible enough that one can use directly the arithmetic data type and Options 
 *   to be used as an \c Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Options> or one can directly
 *   pass only the \c Eigen::Matrix as template parameters
 * 
 */
template <typename T, int Options = Eigen::RowMajor>
using EMapTypeConst = Eigen::Map<typename std::conditional<std::is_arithmetic<T>::value, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Options>, T>::type const>;

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen row vector map type is a new type, used to map the existing C++ memory buffer to an Eigen RowMajor Vector object
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam DataType Data type
 */
template <typename DataType>
using ERowVectorMapType = Eigen::Map<ERowVectorX<DataType>>;

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen row vector constant map type is a new read-only map type to map the existing C++ memory buffer to an Eigen RowMajor Vector object
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam DataType Data type
 */
template <typename DataType>
using ERowVectorMapTypeConst = Eigen::Map<ERowVectorX<DataType> const>;

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen vector map type is a new type to map the existing C++ memory buffer to an Eigen Vector object
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam DataType Data type
 */
template <typename DataType>
using EVectorMapType = Eigen::Map<EVectorX<DataType>>;

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief New read-only map type is used to map the existing C++ memory buffer to an Eigen Vector object
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam DataType Data type
 */
template <typename DataType>
using EVectorMapTypeConst = Eigen::Map<EVectorX<DataType> const>;

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 *  
 * \tparam EigenMatrixType  Eigen matrix type (dynamic_size_storage matrix)
 * 
 * \param  dataPtr  Pointer to the array of data
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * 
 * \returns  Eigen Matrix representation of the array    
 * 
 * \note
 * - If the \c EigenMatrixType template class is a dynamic_size_storage Eigen::Matrix, then one should 
 * provide the number of rows and number of columns at input
 * 
 */
template <class EigenMatrixType>
inline std::enable_if_t<EigenMatrixType::MaxRowsAtCompileTime == Eigen::Dynamic || EigenMatrixType::MaxColsAtCompileTime == Eigen::Dynamic, EigenMatrixType>
EMap(typename EigenMatrixType::Scalar *dataPtr, int const nRows, int const nCols)
{
    return EMapTypeConst<EigenMatrixType>(dataPtr, nRows, nCols);
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 * 
 * \tparam EigenMatrixType Eigen matrix type (fixed_size_storage matrix)
 * 
 * \param dataPtr  Pointer to the array of data
 * 
 * \returns EigenMatrixType  Eigen Matrix representation of the array
 * 
 * \note
 * - If the \c EigenMatrixType template class is a fixed_size_storage \c Eigen::Matrix, then one should not
 * provide the number of rows and number of columns at input
 */
template <class EigenMatrixType>
inline EigenMatrixType EMap(typename EigenMatrixType::Scalar *dataPtr)
{
    return EMapTypeConst<EigenMatrixType>(dataPtr);
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 * Eigen map function copies the existing C++ memory buffer to a temporary Eigen Matrix object 
 * of size(nRows, nCols). 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam EigenMatrixType Eigen matrix type (dynamic_size_storage matrix)
 * 
 * \param  dataPtr  Pointer to the array of data
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * 
 * \returns Eigen Matrix representation of the array    
 *
 * \note
 * - If the \c EigenMatrixType template class is a dynamic_size_storage \c Eigen::Matrix, then the size does must  
 * be passed to the constructor, because it is not specified by the Matrix type.
 */
template <class EigenMatrixType>
inline std::enable_if_t<EigenMatrixType::MaxRowsAtCompileTime == Eigen::Dynamic || EigenMatrixType::MaxColsAtCompileTime == Eigen::Dynamic, EigenMatrixType>
EMap(typename EigenMatrixType::Scalar **dataPtr, int const nRows, int const nCols)
{
    // We have a dynamic_size_storage matrix and it should get the size from number of rows and columns on input
    EigenMatrixType tmpMatrix(nRows, nCols);
    for (int i = 0; i < nRows; i++)
    {
        tmpMatrix.row(i) = EVectorMapTypeConst<typename EigenMatrixType::Scalar>(&dataPtr[i][0], nCols);
    }
    return tmpMatrix;
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen map returns the Eigen Matrix representation of the array from array of data
 * Eigen map function copies the existing C++ memory buffer to a temporary Eigen Matrix object 
 * of size(nRows, nCols). 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 * 
 * \tparam EigenMatrixType Eigen matrix type (fixed_size_storage matrix)
 * 
 * \param dataPtr  Pointer to the array of data
 * 
 * \returns  Eigen Matrix representation of the array    
 *
 * \note
 * - If the \c EigenMatrixType template class is a fixed_size_storage \c Eigen::Matrix, then the size does not have 
 * to be passed to the constructor, because it is already specified by the Matrix type.
 */
template <class EigenMatrixType>
inline EigenMatrixType EMap(typename EigenMatrixType::Scalar **dataPtr)
{
    // We have a fixed_size_storage matrix
    EigenMatrixType tmpMatrix;
    auto nCols = tmpMatrix.cols();
    for (auto i = 0; i < tmpMatrix.rows(); i++)
    {
        tmpMatrix.row(i) = EVectorMapTypeConst<typename EigenMatrixType::Scalar>(&dataPtr[i][0], nCols);
    }
    return tmpMatrix;
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \todo
 * We should add the arraywrapper with inner and outer stride to not copy the data when it is not required
 */

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen map function copies the Eigen Matrix data to the array of data
 * Eigen map function copies the existing Eigen Matrix object to a C++ memory buffer of 
 * the same size as Eigen matrix.
 * 
 * \tparam EigenMatrixType Eigen matrix type
 * 
 * \param dataPtr  Pointer to the array of the same Eigen matrix element type with the same size 
 *                 The data from eMatrix are copied to dataPtr in a rowmajor
 * \param eMatrix  Eigen matrix
 * 
 * \note
 * - We have to copy the data as we do not know before hand that the internal Eigen matrix data 
 * pointer is Aligned, or Unaligned and what is the StrideType
 */
template <class EigenMatrixType>
inline void EMap(typename EigenMatrixType::Scalar *dataPtr, EigenMatrixType const &eMatrix)
{
    EMapType<typename EigenMatrixType::Scalar>(dataPtr, eMatrix.rows(), eMatrix.cols()) = eMatrix;
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen map function copies the Eigen Matrix data to the array of data
 * Eigen map function copies the existing Eigen Matrix object to a C++ memory buffer of 
 * the same size as Eigen matrix.
 * 
 * \tparam EigenMatrixType Eigen matrix type
 * 
 * \param dataPtr  Pointer to the array of the same Eigen matrix element type with the same size 
 *                 The data from eMatrix are copied to dataPtr in a rowmajor
 * \param eMatrix  Eigen matrix
 * 
 * \note
 * - We have to copy the data as we do not know before hand that the internal Eigen matrix data 
 * pointer is Aligned, or Unaligned and what is the StrideType
 */
template <class EigenMatrixType>
inline void EMap(typename EigenMatrixType::Scalar **dataPtr, EigenMatrixType const &eMatrix)
{
    for (auto i = 0; i < eMatrix.rows(); i++)
    {
        EVectorMapType<typename EigenMatrixType::Scalar>(&dataPtr[i][0], eMatrix.cols()) = eMatrix.row(i);
    }
}

/*! 
 * \ingroup LinearAlgebra_Module
 * 
 * \brief This is a check to see if a selfadjoint matrix is positive definite or not?
 * 
 * A matrix is selfadjoint if it equals its adjoint. For real matrices, this means 
 * that the matrix is symmetric: it equals its transpose.
 * 
 * \tparam DataType Data type
 * 
 * \param dataPtr Pointer to the array of data (nDim * nDim)
 * \param nDim    Dimension of a square matrix  
 * 
 * \returns true  If the selfadjoint matrix is positive definite
 * \returns false If the selfadjoint matrix is not positive definite
 */
template <typename DataType>
inline bool isSelfAdjointMatrixPositiveDefinite(DataType *dataPtr, int const nDim)
{
    // First Map the data to Eigen Matrix format
    EMapType<DataType, Eigen::ColMajor> DMap(dataPtr, nDim, nDim);
    // Since the matrix is selfadjoint
    Eigen::SelfAdjointEigenSolver<EMatrixX<DataType>> es(DMap, Eigen::EigenvaluesOnly);

    return es.eigenvalues()(0) > 0 && es.eigenvalues()(0) / es.eigenvalues()(nDim - 1) > machinePrecision<DataType>;
}

/*! 
 * \ingroup LinearAlgebra_Module
 * 
 * \brief This is a check to see if a selfadjoint matrix is positive definite or not?
 * 
 * \tparam EigenMatrixType Eigen Matrix type
 * 
 * \param eMatrix Input matrix
 * 
 * \returns true  If the selfadjoint matrix is positive definite
 * \returns false If the selfadjoint matrix is not positive definite
 */
template <class EigenMatrixType>
inline bool isSelfAdjointMatrixPositiveDefinite(EigenMatrixType const &eMatrix)
{
    // Since the matrix is selfadjoint
    Eigen::SelfAdjointEigenSolver<EigenMatrixType> es(eMatrix, Eigen::EigenvaluesOnly);

    return es.eigenvalues()(0) > 0 && es.eigenvalues()(0) / es.eigenvalues()(eMatrix.rows() - 1) > machinePrecision<typename EigenMatrixType::Scalar>;
}

/*! 
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Force the selfadjoint matrix to be positive definite   
 * 
 * \tparam DataType Data type
 * 
 * \param dataPtr Pointer to the array of data (nDim * nDim)
 * \param nDim    Dimension of a square matrix 
 */
template <typename DataType>
void forceSelfAdjointMatrixPositiveDefinite(DataType *dataPtr, int const nDim)
{
    // First Map the data to Eigen Matrix format. (Symmetric Matrix)
    EMapType<DataType, Eigen::ColMajor> eMatrix(dataPtr, nDim, nDim);

    // Fixed value to increase
    DataType const increaseRate(0.01);

    // Fixed value to multiply by
    DataType const fixedRate(1.01);

#ifdef DEBUG
    std::cout << "eMatrix=" << eMatrix << std::endl;
    std::size_t iter(0);
#endif

    // Vector for diagonal elements
    auto D = eMatrix.diagonal();

    // Whether we have negative or zero ( < machinePrecision) on the diagonal elements fo the matrix
    bool const belowMachinePrecision = (D.array() <= machinePrecision<DataType>).any();

    if (belowMachinePrecision)
    {
        // Find the maximum absolute element on the diagonal of the matrix
        auto MaxAbsD = D.cwiseAbs().maxCoeff();

        // Find the minimum element
        auto const MinD = D.minCoeff();

        MaxAbsD *= increaseRate;

        // There is a negative component on the diagonal, and we should get rid of it.
        if (MinD < 0)
        {
            MaxAbsD -= MinD;
        }

        if (MaxAbsD < machinePrecision<DataType>)
        {
            MaxAbsD = machinePrecision<DataType>;
        }

        for (int i = 0; i < nDim; i++)
        {
            eMatrix(i, i) += MaxAbsD;
        }
        // Now none of the diagonal elements are below machine precision
    }
    else
    {
        for (int i = 0; i < nDim; i++)
        {
            eMatrix(i, i) *= fixedRate;
        }
    }

    while (!isSelfAdjointMatrixPositiveDefinite<DataType>(dataPtr, nDim))
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
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Force the selfadjoint matrix to be positive definite   
 * 
 * \tparam EigenMatrixType Eigen Matrix type
 * 
 * \param eMatrix Input matrix
 */
template <class EigenMatrixType>
void forceSelfAdjointMatrixPositiveDefinite(EigenMatrixType &eMatrix)
{
    typedef typename EigenMatrixType::Scalar DataType;

    // Fixed value to increase
    DataType const increaseRate(0.01);

    // Fixed value to multiply by
    DataType const fixedRate(1.01);

    // Size of the matrix
    auto const nDim = eMatrix.rows();

#ifdef DEBUG
    std::cout << "eMatrix=" << eMatrix << std::endl;
    std::size_t iter(0);
#endif

    // Vector for diagonal elements
    auto D = eMatrix.diagonal();

    bool const belowMachinePrecision = (D.array() <= machinePrecision<DataType>).any();

    if (belowMachinePrecision)
    {
        // Find the maximum absolute element on the diagonal of the matrix
        DataType MaxAbsD = D.cwiseAbs().maxCoeff();

        // Find the minimum element
        DataType const MinD = D.minCoeff();

        MaxAbsD *= increaseRate;

        // There is a negative component on the diagonal, and we should get rid of it.
        if (MinD < 0)
        {
            MaxAbsD -= MinD;
        }

        if (MaxAbsD < machinePrecision<DataType>)
        {
            MaxAbsD = machinePrecision<DataType>;
        }

        for (Eigen::Index i = 0; i < nDim; i++)
        {
            eMatrix(i, i) += MaxAbsD;
        }
        // Now none of the diagonal elements are below machine precision
    }
    else
    {
        for (Eigen::Index i = 0; i < nDim; i++)
        {
            eMatrix(i, i) *= fixedRate;
        }
    }

    while (!isSelfAdjointMatrixPositiveDefinite<EigenMatrixType>(eMatrix))
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

/*!
 * \brief Calculate the squared L2 distance between the \c columns or \c rows of a matrix
 * 
 * \tparam DataType            Data type
 * \tparam OutputDataType      Data type of the return output result (default is double)
 * \tparam VectorwiseOperation Direction of the vector operations in a matrix. (default is \b #ColWise) 
 *                             \sa umuq::VectorwiseOperation
 * 
 * \param eMatrix  Input matrix to calculate the squared distance between its \c columns or \c rows
 * 
 * \returns EMatrixX<OutputDataType> The output matrix to hold the results of the calculation
 *
 */
template <typename DataType, typename OutputDataType = double, VectorwiseOperation Direction = VectorwiseOperation::ColWise>
EMatrixX<OutputDataType> squaredL2Distance(EMatrixX<DataType> const &eMatrix)
{
    switch (Direction)
    {
    case VectorwiseOperation::ColWise:
    {
        auto const nCols = eMatrix.cols();
        EMatrixX<OutputDataType> columnsSquaredDistanceMatrix(nCols, nCols);
        ERowVectorX<OutputDataType> vecC = eMatrix.array().square().colwise().sum().template cast<OutputDataType>();
        columnsSquaredDistanceMatrix.rowwise() = vecC;
        columnsSquaredDistanceMatrix.colwise() += vecC.transpose();
        columnsSquaredDistanceMatrix -= 2 * eMatrix.transpose().template cast<OutputDataType>() * eMatrix.template cast<OutputDataType>();
        return columnsSquaredDistanceMatrix;
    }
    break;
    case VectorwiseOperation::RowWise:
    {
        auto const nRows = eMatrix.rows();
        EMatrixX<OutputDataType> rowsSquaredDistanceMatrix(nRows, nRows);
        EVectorX<OutputDataType> vecR = eMatrix.array().square().rowwise().sum().template cast<OutputDataType>();
        rowsSquaredDistanceMatrix.colwise() = vecR;
        rowsSquaredDistanceMatrix.rowwise() += vecR.transpose();
        rowsSquaredDistanceMatrix -= 2 * eMatrix.template cast<OutputDataType>() * eMatrix.transpose().template cast<OutputDataType>();
        return rowsSquaredDistanceMatrix;
    }
    break;
    default:
        UMUQFAIL("Unknown direction!");
        break;
    };
    /*!
     * \todo
     * Since we do casting at 3 places, it is necessary to do profiling to make sure of the efficiency.
     * Profiling is needed to decide whether we should copy cast the eMatrix to the new matrix first or 
     * continue the current procedure and do casting in operations
     */
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the L2 distance between the \c columns or \c rows of a matrix
 * 
 * \tparam DataType            Data type
 * \tparam OutputDataType      Data type of the return output result (default is double)
 * \tparam VectorwiseOperation Direction of the vector operations in a matrix. (default is \b #ColWise) 
 *                             \sa umuq::VectorwiseOperation
 * 
 * \param eMatrix  The input matrix to calculate the distance between its \c columns or \c rows
 * 
 * \returns EMatrixX<OutputDataType> The output matrix to hold the results of the calculation
 */
template <typename DataType, typename OutputDataType = double, VectorwiseOperation Direction = VectorwiseOperation::ColWise>
EMatrixX<OutputDataType> L2Distance(EMatrixX<DataType> const &eMatrix)
{
    EMatrixX<OutputDataType> dMatrix = squaredL2Distance<DataType, OutputDataType, Direction>(eMatrix);
    return dMatrix.cwiseSqrt();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the squared distance between one point (as an input vector) and a set of points (as an input matrix)
 * 
 * \tparam DataType  Input data type
 * 
 * \param eVector  The input point (input vector) that we want to compute its distance from a set of points
 * \param eMatrix  The set of points as an input matrix where we want to calculate the squared distance between input point and matrix columns
 * 
 * \returns EVectorX<DataType> Vector of squared distances between point and a set of points
 */
template <typename DataType>
inline EVectorX<DataType> squaredL2Distance(EVectorX<DataType> const &eVector, EMatrixX<DataType> const &eMatrix)
{
    EVectorX<DataType> dVector(eMatrix.cols());
    dVector.fill(eVector.squaredNorm());
    return (dVector + eMatrix.cwiseProduct(eMatrix).colwise().sum().transpose() - 2 * eMatrix.transpose() * eVector).cwiseAbs();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the distance between one point (as an input vector) and a set of points (as an input matrix)
 * 
 * \tparam DataType  Input data type
 * 
 * \param eVector  The input point (input vector) that we want to compute its distance from a set of points
 * \param eMatrix  The set of points as an input matrix where we want to calculate the distance between input point and matrix columns
 * 
 * \returns EVectorX<DataType> Vector of distances between point and a set of points
 */
template <typename DataType>
inline EVectorX<DataType> L2Distance(EVectorX<DataType> const &eVector, EMatrixX<DataType> const &eMatrix)
{
    EVectorX<DataType> dVector = squaredL2Distance<DataType>(eVector, eMatrix);
    return dVector.cwiseSqrt();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the squared distance between set of points (as an input matrix) and a set of points (as a second input matrix)
 * 
 * \tparam DataType  Input data type
 * 
 * \param eMatrix1  The set of point (input matrix) that we want to compute it distances from a set of points
 * \param eMatrix2  The set of points as an input matrix where we want to calculate the squared distance between input point and matrix columns
 * 
 * \returns EMatrixX<DataType> Vector of squared distances between set of points and another set of points
 */
template <typename DataType>
inline EMatrixX<DataType> squaredL2Distance(EMatrixX<DataType> const &eMatrix1, EMatrixX<DataType> const &eMatrix2)
{
    EMatrixX<DataType> dMatrix = -2 * (eMatrix1.transpose() * eMatrix2);
    dMatrix.rowwise() += eMatrix2.cwiseProduct(eMatrix2).colwise().sum();
    dMatrix.colwise() += eMatrix1.cwiseProduct(eMatrix1).colwise().sum().transpose();
    return dMatrix.cwiseAbs();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the distance between set of points (as an input matrix) and a set of points (as a second input matrix)
 * 
 * \tparam DataType  Input data type
 * 
 * \param eMatrix1  The set of point (input matrix) that we want to compute it distances from a set of points
 * \param eMatrix2  The set of points as an input matrix where we want to calculate the squared distance between input point and matrix columns
 * 
 * \returns EMatrixX<DataType> Vector of squared distances between set of points and another set of points
 */
template <typename DataType>
inline EMatrixX<DataType> L2Distance(EMatrixX<DataType> const &eMatrix1, EMatrixX<DataType> const &eMatrix2)
{
    EMatrixX<DataType> dMatrix = -2 * (eMatrix1.transpose() * eMatrix2);
    dMatrix.rowwise() += eMatrix2.cwiseProduct(eMatrix2).colwise().sum();
    dMatrix.colwise() += eMatrix1.cwiseProduct(eMatrix1).colwise().sum().transpose();
    return dMatrix.cwiseAbs().cwiseSqrt();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the S-optimality measure
 * 
 * \tparam DataType            Input data type
 * \tparam OutputDataType      Output data type (default is double)
 * \tparam VectorwiseOperation Direction of the vector operations in a matrix. (default is \b #ColWise) 
 *                             \sa umuq::VectorwiseOperation
 * 
 * \param eMatrix The matrix to calculate its column or row wise S-optimality 
 * 
 * \returns OutputDataType The S-optimality measure
 * 
 * The S-optimality measure: <br>
 * The S-optimality is presented by L{\"a}uter (1974). It aims to maximize the harmonic mean distance from 
 * each design point to all the other points in the design. Mathematically, an S–optimal design maximizes 
 * \f$ \frac{N_D}{ \sum_{y \in D} {1/d(y, D-y)}}. \f$ where D is the set of design points, and \f$ N_D \f$ 
 * is the number of points in \f$ D \f$, the distances \f$ d(y, D-y) \f$ are large, so the points are as 
 * spread out as possible. This measures how spread out the design points are; therefore, an S–optimal 
 * design is also called a maximum spread design. 
 * 
 * Reference: <br>
 * E. L{\"a}uter, "Experimental design in a class of models," Optimization: A Journal of 
 * Mathematical Programming and Operations Research, 5 (1964), p. 379--398
 */
template <typename DataType, typename OutputDataType = double, VectorwiseOperation Direction = VectorwiseOperation::ColWise>
OutputDataType SOptimality(EMatrixX<DataType> const &eMatrix)
{
    EMatrixX<OutputDataType> dMatrix;
    dMatrix = L2Distance<DataType, OutputDataType, Direction>(eMatrix);
    return 1. / dMatrix.cwiseInverse().sum();
}

/*!
 * \brief Map a vector in space \f$ [low, high]^n \f$ to the unit box \f$ [0, 1]^n \f$
 * 
 * \tparam RealType Floating point data type
 * 
 * \param eVector  Input vector      
 * \param low      Vector of lower bounds
 * \param high     Vector of upper bounds
 * 
 * \returns EVectorX<RealType> Scaled vector mapped to the unit box
 */
template <typename RealType>
inline EVectorX<RealType> scaleToUnitBox(const EVectorX<RealType> &eVector, const EVectorX<RealType> &low, const EVectorX<RealType> &high)
{
    return (eVector - low).cwiseProduct((high - low).cwiseInverse());
};

/*!
 * \brief Map a matrix from a space of \f$ [low, high]^n \f$ to the unit box of \f$ [0, 1]^n \f$
 * 
 * \tparam RealType Floating point data type
 * 
 * \param eMatrix  Input matrix      
 * \param low      Vector of lower bounds
 * \param high     Vector of upper bounds 
 * 
 * \returns EMatrixX<RealType> Scaled matrix mapped to the unit box
 */
template <typename RealType>
inline EMatrixX<RealType> scaleToUnitBox(const EMatrixX<RealType> &eMatrix, const EVectorX<RealType> &low, const EVectorX<RealType> &high)
{
    return (eMatrix.colwise() - low).array().colwise() * (high - low).cwiseInverse().array();
};

/*!
 * \brief Map a vector from the unit box space \f$ [0, 1]^n \f$ to the hypercube space of \f$ [low, high]^n \f$ 
 *  
 * \tparam RealType Floating point data type
 * 
 * \param eVector  Input vector      
 * \param low      Vector of lower bounds
 * \param high     Vector of upper bounds
 * 
 * \returns EVectorX<RealType> Scaled vector mapped to the hypercube space
 */
template <typename RealType>
inline EVectorX<RealType> scaleToHyperCube(const EVectorX<RealType> &eVector, const EVectorX<RealType> &low, const EVectorX<RealType> &high)
{
    return low + (high - low).cwiseProduct(eVector);
};

/*!
 * \brief Map a matrix (multiple points) from the unit box space \f$ [0, 1]^n \f$ to the hypercube space of \f$ [low, high]^n \f$ 
 *  
 * \tparam RealType Floating point data type
 * 
 * \param eVector  Input vector      
 * \param low      Vector of lower bounds
 * \param high     Vector of upper bounds
 * 
 * \returns  EMatrixX<RealType> Scaled matrix mapped the hypercube space
 */
template <typename RealType>
inline EMatrixX<RealType> scaleToHyperCube(const EMatrixX<RealType> &eMatrix, const EVectorX<RealType> &low, const EVectorX<RealType> &high)
{
    return (eMatrix.array().colwise() * (high - low).array()).matrix().colwise() + low;
};

} // namespace linearalgebra
} // namespace umuq

#endif // UMUQ_EIGENLIB
