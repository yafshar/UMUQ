#ifndef UMHBM_EIGENMATRIX_H
#define UMHBM_EIGENMATRIX_H

#include <Eigen/Dense>

// Standard typedef from eigen
typedef typename Eigen::Matrix<double, 2, 2> EMatrix2d;
typedef typename Eigen::Matrix<double, 2, Eigen::Dynamic> EMatrix2Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 2> EMatrixX2d;

typedef typename Eigen::Matrix<double, 3, 3> EMatrix3d;
typedef typename Eigen::Matrix<double, 3, Eigen::Dynamic> EMatrix3Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 3> EMatrixX3d;

typedef typename Eigen::Matrix<double, 4, 4> EMatrix4d;
typedef typename Eigen::Matrix<double, 4, Eigen::Dynamic> EMatrix4Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 4> EMatrixX4d;

typedef typename Eigen::Matrix<double, 5, 5> EMatrix5d;
typedef typename Eigen::Matrix<double, 5, Eigen::Dynamic> EMatrix5Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 5> EMatrixX5d;

typedef typename Eigen::Matrix<double, 6, 6> EMatrix6d;
typedef typename Eigen::Matrix<double, 6, Eigen::Dynamic> EMatrix6Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 6> EMatrixX6d;

typedef typename Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EMatrixXd;

typedef typename Eigen::Matrix<float, 2, 2> EMatrix2f;
typedef typename Eigen::Matrix<float, 2, Eigen::Dynamic> EMatrix2Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 2> EMatrixX2f;

typedef typename Eigen::Matrix<float, 3, 3> EMatrix3f;
typedef typename Eigen::Matrix<float, 3, Eigen::Dynamic> EMatrix3Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 3> EMatrixX3f;

typedef typename Eigen::Matrix<float, 4, 4> EMatrix4f;
typedef typename Eigen::Matrix<float, 4, Eigen::Dynamic> EMatrix4Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 4> EMatrixX4f;

typedef typename Eigen::Matrix<float, 5, 5> EMatrix5f;
typedef typename Eigen::Matrix<float, 5, Eigen::Dynamic> EMatrix5Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 5> EMatrixX5f;

typedef typename Eigen::Matrix<float, 6, 6> EMatrix6f;
typedef typename Eigen::Matrix<float, 6, Eigen::Dynamic> EMatrix6Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 6> EMatrixX6f;

typedef typename Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> EMatrixXf;

typedef typename Eigen::Matrix<int, 2, 2> EMatrix2i;
typedef typename Eigen::Matrix<int, 2, Eigen::Dynamic> EMatrix2Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 2> EMatrixX2i;

typedef typename Eigen::Matrix<int, 3, 3> EMatrix3i;
typedef typename Eigen::Matrix<int, 3, Eigen::Dynamic> EMatrix3Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 3> EMatrixX3i;

typedef typename Eigen::Matrix<int, 4, 4> EMatrix4i;
typedef typename Eigen::Matrix<int, 4, Eigen::Dynamic> EMatrix4Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 4> EMatrixX4i;

typedef typename Eigen::Matrix<int, 5, 5> EMatrix5i;
typedef typename Eigen::Matrix<int, 5, Eigen::Dynamic> EMatrix5Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 5> EMatrixX5i;

typedef typename Eigen::Matrix<int, 6, 6> EMatrix6i;
typedef typename Eigen::Matrix<int, 6, Eigen::Dynamic> EMatrix6Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 6> EMatrixX6i;

typedef typename Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> EMatrixXi;

typedef typename Eigen::Matrix<double, 1, 2> ERowVector2d;
typedef typename Eigen::Matrix<double, 1, 3> ERowVector3d;
typedef typename Eigen::Matrix<double, 1, 4> ERowVector4d;
typedef typename Eigen::Matrix<double, 1, 5> ERowVector5d;
typedef typename Eigen::Matrix<double, 1, 6> ERowVector6d;
typedef typename Eigen::Matrix<double, 1, Eigen::Dynamic> ERowVectorXd;

typedef typename Eigen::Matrix<float, 1, 2> ERowVector2f;
typedef typename Eigen::Matrix<float, 1, 3> ERowVector3f;
typedef typename Eigen::Matrix<float, 1, 4> ERowVector4f;
typedef typename Eigen::Matrix<float, 1, 5> ERowVector5f;
typedef typename Eigen::Matrix<float, 1, 6> ERowVector6f;
typedef typename Eigen::Matrix<float, 1, Eigen::Dynamic> ERowVectorXf;

typedef typename Eigen::Matrix<int, 1, 2> ERowVector2i;
typedef typename Eigen::Matrix<int, 1, 3> ERowVector3i;
typedef typename Eigen::Matrix<int, 1, 4> ERowVector4i;
typedef typename Eigen::Matrix<int, 1, 5> ERowVector5i;
typedef typename Eigen::Matrix<int, 1, 6> ERowVector6i;
typedef typename Eigen::Matrix<int, 1, Eigen::Dynamic> ERowVectorXi;

typedef typename Eigen::Matrix<double, 2, 1> EVector2d;
typedef typename Eigen::Matrix<double, 3, 1> EVector3d;
typedef typename Eigen::Matrix<double, 4, 1> EVector4d;
typedef typename Eigen::Matrix<double, 5, 1> EVector5d;
typedef typename Eigen::Matrix<double, 6, 1> EVector6d;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 1> EVectorXd;

typedef typename Eigen::Matrix<float, 2, 1> EVector2f;
typedef typename Eigen::Matrix<float, 3, 1> EVector3f;
typedef typename Eigen::Matrix<float, 4, 1> EVector4f;
typedef typename Eigen::Matrix<float, 5, 1> EVector5f;
typedef typename Eigen::Matrix<float, 6, 1> EVector6f;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 1> EVectorXf;

typedef typename Eigen::Matrix<int, 2, 1> EVector2i;
typedef typename Eigen::Matrix<int, 3, 1> EVector3i;
typedef typename Eigen::Matrix<int, 4, 1> EVector4i;
typedef typename Eigen::Matrix<int, 5, 1> EVector5i;
typedef typename Eigen::Matrix<int, 6, 1> EVector6i;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 1> EVectorXi;

template <typename T>
using EVectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using ERowVectorX = Eigen::Matrix<T, 1, Eigen::Dynamic>;

template <typename T>
using EMatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

/*!
 * \brief Stores a set of parameters controlling the way matrices are printed
 * 
 * - precision \c FullPrecision.
 * - coeffSeparator string printed between two coefficients of the same row
 * - rowSeparator string printed between two rows
 */
Eigen::IOFormat fmt(Eigen::FullPrecision);

/*!
 * \brief New type to map the existing C++ memory buffer to an Eigen Matrix object in a RowMajor
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam TdataPtr typedef of the data pointer    
 * 
 */
template <typename TdataPtr, int _Options = Eigen::RowMajor>
using TEMapX = Eigen::Map<Eigen::Matrix<TdataPtr, Eigen::Dynamic, Eigen::Dynamic, _Options>>;

/*!
 * \brief New a read-only map type to map the existing C++ memory buffer to an Eigen Matrix object in a RowMajor
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam TdataPtr typedef of the data pointer    
 * 
 */
template <typename TdataPtr, int _Options = Eigen::RowMajor>
using CTEMapX = Eigen::Map<const Eigen::Matrix<TdataPtr, Eigen::Dynamic, Eigen::Dynamic, _Options>>;

/*!
 * \brief New type to map the existing C++ memory buffer to an Eigen Matrix object of type double in a RowMajor
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures.  
 * 
 */
using TEMapXd = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

/*!
 * \brief New a read-only map type to map the existing C++ memory buffer to an Eigen Matrix object of type double in a RowMajor
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures.  
 * 
 */
using CTEMapXd = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

/*!
 * \brief EMapX function copies the existing C++ memory buffer to an Eigen Matrix object of size(nRows, nCols) 
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam TEMX     typedef for Eigen matrix 
 * \tparam TdataPtr typedef of the pointer to the array to map 
 * 
 * \param  dataPtr  pointer to the array to map of type TdataPtr
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * \param  EMapX    Eigen Matrix representation of the array    
 * 
 */
template <typename TEMX, typename TdataPtr>
TEMX EMapX(TdataPtr *dataPtr, size_t nRows, size_t nCols)
{
    return Eigen::Map<const TEMX>(dataPtr, nRows, nCols);
}

template <typename TEMX, typename TdataPtr>
TEMX EMapX(TdataPtr **dataPtr, size_t nRows, size_t nCols)
{
    TEMX MTemp(nRows, nCols);

    for (size_t i = 0; i < nRows; i++)
    {
        MTemp.row(i) = Eigen::Matrix<TdataPtr, Eigen::Dynamic, 1>::Map(&dataPtr[i][0], nCols);
    }

    return MTemp;
}

/*!
 * \brief Pointer will now point to a beginning of a memory buffer of the Eigen’s data structures
 *  
 * The Map operation maps the existing Eigen’s data structure to the memory buffer
 * 
 * \tparam TEMX     typedef for Eigen matrix 
 * \tparam TdataPtr typedef of the pointer to the array
 * 
 * \param  EMX     Input Eigen’s matrix of type TEMX
 * \param  dataPtr pointer to the array of type TdataPtr
 */
template <typename TEMX, typename TdataPtr>
void EMapX(const TEMX EMX, TdataPtr *dataPtr)
{
    Eigen::Map<TEMX>(dataPtr, EMX.rows(), EMX.cols()) = EMX;
}

template <typename TEMX, typename TdataPtr>
void EMapX(const TEMX EMX, TdataPtr **dataPtr)
{
    for (size_t i = 0; i < EMX.rows(); i++)
    {
        Eigen::Matrix<TdataPtr, Eigen::Dynamic, 1>::Map(&dataPtr[i][0], EMX.cols()) = EMX.row(i);
    }
}

/*!
 * \brief Copy the existing pointer to the C++ memory buffer of type double to an Eigen object
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures.  
 * 
 * \param  dataPtr  pointer to the array to map of type double
 * \param  nRows    Number of Rows
 * \param  nCols    Number of Columns
 * \param  EMapXd   Eigen Matrix representation of data
 */
EMatrixXd EMapXd(double *dataPtr, size_t nRows, size_t nCols)
{
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(dataPtr, nRows, nCols);
}

EMatrixXd EMapXd(double **dataPtr, size_t nRows, size_t nCols)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MTemp(nRows, nCols);

    for (size_t i = 0; i < nRows; i++)
    {
        MTemp.row(i) = EVectorXd::Map(&dataPtr[i][0], nCols);
    }

    return MTemp;
}

/*!
 * \brief Pointer will now point to a beginning of a memory region of the Eigen’s data structures
 *  
 * The Map operation maps the existing Eigen’s data structure to the memory buffer
 * 
 * \param  EMXd    Input Eigen’s matrix of type double
 * \param  dataPtr Pointer to the memory buffer of type double
 */
void EMapXd(EMatrixXd const EMXd, double *dataPtr)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(dataPtr, EMXd.rows(), EMXd.cols()) = EMXd;
}

void EMapXd(EMatrixXd const EMXd, double **dataPtr)
{
    for (size_t i = 0; i < EMXd.rows(); i++)
    {
        EVectorXd::Map(&dataPtr[i][0], EMXd.cols()) = EMXd.row(i);
    }
}

#endif
