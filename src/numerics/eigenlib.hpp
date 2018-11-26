#ifndef UMUQ_EIGENLIB_H
#define UMUQ_EIGENLIB_H

#include "data/eigendatatype.hpp"

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
 * \tparam T         Data type or \b Eigen::Matrix type
 * \tparam _Options  optional parameter, a combination of either 
 *                   - \b Eigen::RowMajor or \b Eigen::ColMajor, <br> 
 *                     or one of either <br>
 *                   - \b Eigen::AutoAlign, or \b Eigen::DontAlign. <br>
 *                   The former controls storage order, and defaults to column-major. The latter controls alignment, which is required
 *                   for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
 *
 * 
 * \note 
 * 	- Use of template is flexible enough that one can use directly the arithmetic data type and _Options 
 *    to be used as an \c Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> or one can directly
 *    pass only the \c Eigen::Matrix as template parameters
 * 
 * Example: <br>
 * Simply mapping a contiguous C++ memory buffer as a column-major Eigen Matrix object:
 * \code 
 * double A[12];
 * for(int i = 0; i < 12; ++i) A[i] = (double)i;
 * EMapType<double, Eigen::ColMajor> B(A, 3, 4); 
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
 * EMapType<double> B(A, 3, 4); 
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
template <class T, int _Options = Eigen::RowMajor>
using EMapType = Eigen::Map<typename std::conditional<std::is_arithmetic<T>::value, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options>, T>::type>;

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Eigen map type constant is a new read-only map type to map the existing C++ memory buffer to an Eigen Matrix object 
 * The Map operation maps the existing memory region into the Eigen’s data structures.
 * 
 * \tparam T         Data type or \b Eigen::Matrix type
 * \tparam _Options  optional parameter, a combination of either 
 *                   - \b Eigen::RowMajor or \b Eigen::ColMajor, <br>
 *                     or one of either <br>
 *                   - \b Eigen::AutoAlign or \b Eigen::DontAlign. <br>
 *                   The former controls storage order, and defaults to column-major. The latter controls alignment, which is required
 *                   for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
 * 
 * \note 
 * - Use of template is flexible enough that one can use directly the arithmetic data type and _Options 
 *   to be used as an \c Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> or one can directly
 *   pass only the \c Eigen::Matrix as template parameters
 * 
 */
template <typename T, int _Options = Eigen::RowMajor>
using EMapTypeConst = Eigen::Map<typename std::conditional<std::is_arithmetic<T>::value, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options>, T>::type const>;

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
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the squared distance between the rows of a matrix
 * 
 * \tparam DataType       Data type
 * \tparam OutputDataType Data type of the return output result (default is double)
 *
 * \param inMatrix   The input matrix to calculate the squared distance between its rows
 * \param outMatrix  The output matrix to hold the results of the calculation
 */
template <typename DataType, typename OutputDataType = double>
void calculateRowsSquaredDistance(EMatrixX<DataType> const &inMatrix, EMatrixX<OutputDataType> &outMatrix)
{
    auto const nRows = inMatrix.rows();
    if (outMatrix.rows() != nRows || outMatrix.cols() != nRows)
    {
        outMatrix.resize(nRows, nRows);
    }
    /*!
     * \todo
     * Since we do casting at 3 places, it is necessary to do profiling to make sure of the efficiency.
     * Profiling is needed to decide whether we should copy cast the inMatrix to the new matrix first or 
     * continue the current procedure and do casting in operations
     */
    EVectorX<OutputDataType> vecX = inMatrix.array().square().rowwise().sum().template cast<OutputDataType>();
    outMatrix.colwise() = vecX;
    outMatrix.rowwise() += vecX.transpose();
    outMatrix -= 2 * inMatrix.template cast<OutputDataType>() * inMatrix.transpose().template cast<OutputDataType>();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the squared distance between the columns of a matrix
 * 
 * \tparam DataType       Input data type
 * \tparam OutputDataType Output data type (default is double)
 *
 * \param inMatrix   The input matrix to calculate the squared distance between its columns
 * \param outMatrix  The output matrix to hold the results of the calculation
 */
template <typename DataType, typename OutputDataType = double>
void calculateColumnsSquaredDistance(EMatrixX<DataType> const &inMatrix, EMatrixX<OutputDataType> &outMatrix)
{
    auto const nCols = inMatrix.cols();
    if (outMatrix.rows() != nCols || outMatrix.cols() != nCols)
    {
        outMatrix.resize(nCols, nCols);
    }
    ERowVectorX<OutputDataType> vecX = inMatrix.array().square().colwise().sum().template cast<OutputDataType>();
    outMatrix.rowwise() = vecX;
    outMatrix.colwise() += vecX.transpose();
    outMatrix -= 2 * inMatrix.transpose().template cast<OutputDataType>() * inMatrix.template cast<OutputDataType>();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the distance between the rows of a matrix
 * 
 * \tparam DataType       Input data type
 * \tparam OutputDataType Output data type (default is double)
 * 
 * \param inMatrix   The input matrix to calculate the distance between its rows
 * \param outMatrix  The output matrix to hold the results of the calculation
 */
template <typename DataType, typename OutputDataType = double>
void calculateRowsDistance(EMatrixX<DataType> const &inMatrix, EMatrixX<OutputDataType> &outMatrix)
{
    calculateRowsSquaredDistance<DataType, OutputDataType>(inMatrix, outMatrix);
    outMatrix = outMatrix.cwiseSqrt();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the distance between the columns of a matrix
 * 
 * \tparam DataType       Input data type
 * \tparam OutputDataType Output data type (default is double)
 *
 * \param inMatrix   The input matrix to calculate the distance between its columns
 * \param outMatrix  The output matrix to hold the results of the calculation
 */
template <typename DataType, typename OutputDataType = double>
void calculateColumnsDistance(EMatrixX<DataType> const &inMatrix, EMatrixX<OutputDataType> &outMatrix)
{
    calculateColumnsSquaredDistance<DataType, OutputDataType>(inMatrix, outMatrix);
    outMatrix = outMatrix.cwiseSqrt();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the S-optimality measure
 * 
 * \tparam DataType       Input data type
 * \tparam OutputDataType Output data type (default is double)
 * 
 * \param inMatrix The matrix to calculate its row wise S-optimality 
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
template <typename DataType, typename OutputDataType = double>
OutputDataType calculateRowWiseSOptimality(EMatrixX<DataType> const &inMatrix)
{
    EMatrixX<OutputDataType> outMatrix;
    calculateRowsDistance<DataType, OutputDataType>(inMatrix, outMatrix);
    return 1. / outMatrix.cwiseInverse().sum();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Calculate the S-optimality measure
 * 
 * \tparam DataType       Input data type
 * \tparam OutputDataType Output data type (default is double)
 * 
 * \param inMatrix The matrix to calculate its row wise S-optimality 
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
template <typename DataType, typename OutputDataType = double>
OutputDataType calculateColumnWiseSOptimality(EMatrixX<DataType> const &inMatrix)
{
    EMatrixX<OutputDataType> outMatrix;
    calculateColumnsDistance<DataType, OutputDataType>(inMatrix, outMatrix);
    return 1. / outMatrix.cwiseInverse().sum();
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Fills the rows of the matrix with the specified size whose coefficients are equally spaced between low and high.
 * 
 * The function generates equally spaced values in each row of the matrix with the closed interval of \f$ [low, high].\f$
 * 
 * \tparam DataType Input data type
 * 
 * \param eMatrix  The Eigen matrix
 * \param low      Lower bound of the interval
 * \param high     Upperbound of the interval
 */
template <typename DataType>
void rowWiseLinSpaced(EMatrixX<DataType> &eMatrix, DataType const low, DataType const high)
{
    ERowVectorX<DataType> const row = ERowVectorX<DataType>::LinSpaced(eMatrix.cols(), low, high);
    eMatrix.rowwise() = row;
}

/*!
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Fills the columns of the matrix with the specified size whose coefficients are equally spaced between low and high.
 * 
 * The function generates equally spaced values in each column of the matrix with the closed interval of \f$ [low, high].\f$
 * 
 * \tparam DataType Input data type
 * 
 * \param eMatrix  The Eigen matrix
 * \param low      Lower bound of the interval
 * \param high     Upperbound of the interval
 */
template <typename DataType>
void columnWiseLinSpaced(EMatrixX<DataType> &eMatrix, DataType const low, DataType const high)
{
    EVectorX<DataType> const col = EVectorX<DataType>::LinSpaced(eMatrix.rows(), low, high);
    eMatrix.colwise() = col;
}

} // namespace linearalgebra
} // namespace umuq

#endif // UMUQ_EIGENLIB
