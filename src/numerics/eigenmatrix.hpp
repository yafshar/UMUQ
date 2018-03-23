#ifndef UMHBM_EIGENMATRIX_H
#define UMHBM_EIGENMATRIX_H

#include <Eigen/Dense>

// Standard typedef from eigen
typedef Eigen::Matrix<double, 2, 2> EMatrix2d;
typedef Eigen::Matrix<double, 2, Eigen::Dynamic> EMatrix2Xd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 2> EMatrixX2d;

typedef Eigen::Matrix<double, 3, 3> EMatrix3d;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic> EMatrix3Xd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 3> EMatrixX3d;

typedef Eigen::Matrix<double, 4, 4> EMatrix4d;
typedef Eigen::Matrix<double, 4, Eigen::Dynamic> EMatrix4Xd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 4> EMatrixX4d;

typedef Eigen::Matrix<double, 5, 5> EMatrix5d;
typedef Eigen::Matrix<double, 5, Eigen::Dynamic> EMatrix5Xd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 5> EMatrixX5d;

typedef Eigen::Matrix<double, 6, 6> EMatrix6d;
typedef Eigen::Matrix<double, 6, Eigen::Dynamic> EMatrix6Xd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 6> EMatrixX6d;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EMatrixXd;

typedef Eigen::Matrix<float, 2, 2> EMatrix2f;
typedef Eigen::Matrix<float, 2, Eigen::Dynamic> EMatrix2Xf;
typedef Eigen::Matrix<float, Eigen::Dynamic, 2> EMatrixX2f;

typedef Eigen::Matrix<float, 3, 3> EMatrix3f;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic> EMatrix3Xf;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3> EMatrixX3f;

typedef Eigen::Matrix<float, 4, 4> EMatrix4f;
typedef Eigen::Matrix<float, 4, Eigen::Dynamic> EMatrix4Xf;
typedef Eigen::Matrix<float, Eigen::Dynamic, 4> EMatrixX4f;

typedef Eigen::Matrix<float, 5, 5> EMatrix5f;
typedef Eigen::Matrix<float, 5, Eigen::Dynamic> EMatrix5Xf;
typedef Eigen::Matrix<float, Eigen::Dynamic, 5> EMatrixX5f;

typedef Eigen::Matrix<float, 6, 6> EMatrix6f;
typedef Eigen::Matrix<float, 6, Eigen::Dynamic> EMatrix6Xf;
typedef Eigen::Matrix<float, Eigen::Dynamic, 6> EMatrixX6f;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> EMatrixXf;

typedef Eigen::Matrix<int, 2, 2> EMatrix2i;
typedef Eigen::Matrix<int, 2, Eigen::Dynamic> EMatrix2Xi;
typedef Eigen::Matrix<int, Eigen::Dynamic, 2> EMatrixX2i;

typedef Eigen::Matrix<int, 3, 3> EMatrix3i;
typedef Eigen::Matrix<int, 3, Eigen::Dynamic> EMatrix3Xi;
typedef Eigen::Matrix<int, Eigen::Dynamic, 3> EMatrixX3i;

typedef Eigen::Matrix<int, 4, 4> EMatrix4i;
typedef Eigen::Matrix<int, 4, Eigen::Dynamic> EMatrix4Xi;
typedef Eigen::Matrix<int, Eigen::Dynamic, 4> EMatrixX4i;

typedef Eigen::Matrix<int, 5, 5> EMatrix5i;
typedef Eigen::Matrix<int, 5, Eigen::Dynamic> EMatrix5Xi;
typedef Eigen::Matrix<int, Eigen::Dynamic, 5> EMatrixX5i;

typedef Eigen::Matrix<int, 6, 6> EMatrix6i;
typedef Eigen::Matrix<int, 6, Eigen::Dynamic> EMatrix6Xi;
typedef Eigen::Matrix<int, Eigen::Dynamic, 6> EMatrixX6i;

typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> EMatrixXi;

typedef Eigen::Matrix<double, 1, 2> ERowVector2d;
typedef Eigen::Matrix<double, 1, 3> ERowVector3d;
typedef Eigen::Matrix<double, 1, 4> ERowVector4d;
typedef Eigen::Matrix<double, 1, 5> ERowVector5d;
typedef Eigen::Matrix<double, 1, 6> ERowVector6d;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> ERowVectorXd;

typedef Eigen::Matrix<float, 1, 2> ERowVector2f;
typedef Eigen::Matrix<float, 1, 3> ERowVector3f;
typedef Eigen::Matrix<float, 1, 4> ERowVector4f;
typedef Eigen::Matrix<float, 1, 5> ERowVector5f;
typedef Eigen::Matrix<float, 1, 6> ERowVector6f;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> ERowVectorXf;

typedef Eigen::Matrix<int, 1, 2> ERowVector2i;
typedef Eigen::Matrix<int, 1, 3> ERowVector3i;
typedef Eigen::Matrix<int, 1, 4> ERowVector4i;
typedef Eigen::Matrix<int, 1, 5> ERowVector5i;
typedef Eigen::Matrix<int, 1, 6> ERowVector6i;
typedef Eigen::Matrix<int, 1, Eigen::Dynamic> ERowVectorXi;

typedef Eigen::Matrix<double, 2, 1> EVector2d;
typedef Eigen::Matrix<double, 3, 1> EVector3d;
typedef Eigen::Matrix<double, 4, 1> EVector4d;
typedef Eigen::Matrix<double, 5, 1> EVector5d;
typedef Eigen::Matrix<double, 6, 1> EVector6d;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> EVectorXd;

typedef Eigen::Matrix<float, 2, 1> EVector2f;
typedef Eigen::Matrix<float, 3, 1> EVector3f;
typedef Eigen::Matrix<float, 4, 1> EVector4f;
typedef Eigen::Matrix<float, 5, 1> EVector5f;
typedef Eigen::Matrix<float, 6, 1> EVector6f;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> EVectorXf;

typedef Eigen::Matrix<int, 2, 1> EVector2i;
typedef Eigen::Matrix<int, 3, 1> EVector3i;
typedef Eigen::Matrix<int, 4, 1> EVector4i;
typedef Eigen::Matrix<int, 5, 1> EVector5i;
typedef Eigen::Matrix<int, 6, 1> EVector6i;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> EVectorXi;

/*!
 * \brief Map the existing memory buffer to an Eigen object
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
    return Eigen::Map<TEMX>(dataPtr, nRows, nCols);
}

/*!
 * \brief Pointer will now point to a beginning of a memory region of the Eigen’s data structures
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
void EMapX(TEMX EMX, TdataPtr *dataPtr)
{
    Eigen::Map<TEMX>(dataPtr, EMX.rows(), EMX.cols()) = EMX;
}

/*!
 * \brief Map the existing pointer to the array of type double to an Eigen object
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures.  
 * 
 * \param  dataPtr  pointer to the array to map of type double
 * \param  nRows    Number of Rows
 * \param  nCols    Number of Columns
 * \param  EMapXd   Eigen Matrix representation of data
 */
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EMapXd(double *dataPtr, size_t nRows, size_t nCols)
{
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(dataPtr, nRows, nCols);
}

/*!
 * \brief Pointer will now point to a beginning of a memory region of the Eigen’s data structures
 *  
 * The Map operation maps the existing Eigen’s data structure to the memory buffer
 * 
 * \param  EMXd    Input Eigen’s matrix of type double
 * \param  dataPtr Pointer to the memory buffer of type double
 */
void EMapXd(EMatrixXd EMXd, double *dataPtr)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(dataPtr, EMXd.rows(), EMXd.cols()) = EMXd;
}

#endif
