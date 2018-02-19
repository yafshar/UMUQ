#ifndef EIGENMATRIX_H
#define EIGENMATRIX_H

#include <Eigen/Dense>

template <typename _Scalar, int _Rows, int _Cols>
class EMatrix : public Eigen::Matrix<_Scalar, _Rows, _Cols>
{
};

// Standard typedef from eigen
typedef EMatrix<double, 2, 2> EMatrix2d;
typedef EMatrix<double, 2, Eigen::Dynamic> EMatrix2Xd;
typedef EMatrix<double, Eigen::Dynamic, 2> EMatrixX2d;

typedef EMatrix<double, 3, 3> EMatrix3d;
typedef EMatrix<double, 3, Eigen::Dynamic> EMatrix3Xd;
typedef EMatrix<double, Eigen::Dynamic, 3> EMatrixX3d;

typedef EMatrix<double, 4, 4> EMatrix4d;
typedef EMatrix<double, 4, Eigen::Dynamic> EMatrix4Xd;
typedef EMatrix<double, Eigen::Dynamic, 4> EMatrixX4d;

typedef EMatrix<double, 5, 5> EMatrix5d;
typedef EMatrix<double, 5, Eigen::Dynamic> EMatrix5Xd;
typedef EMatrix<double, Eigen::Dynamic, 5> EMatrixX5d;

typedef EMatrix<double, 6, 6> EMatrix6d;
typedef EMatrix<double, 6, Eigen::Dynamic> EMatrix6Xd;
typedef EMatrix<double, Eigen::Dynamic, 6> EMatrixX6d;

typedef EMatrix<double, Eigen::Dynamic, Eigen::Dynamic> EMatrixXd;

typedef EMatrix<float, 2, 2> EMatrix2f;
typedef EMatrix<float, 2, Eigen::Dynamic> EMatrix2Xf;
typedef EMatrix<float, Eigen::Dynamic, 2> EMatrixX2f;

typedef EMatrix<float, 3, 3> EMatrix3f;
typedef EMatrix<float, 3, Eigen::Dynamic> EMatrix3Xf;
typedef EMatrix<float, Eigen::Dynamic, 3> EMatrixX3f;

typedef EMatrix<float, 4, 4> EMatrix4f;
typedef EMatrix<float, 4, Eigen::Dynamic> EMatrix4Xf;
typedef EMatrix<float, Eigen::Dynamic, 4> EMatrixX4f;

typedef EMatrix<float, 5, 5> EMatrix5f;
typedef EMatrix<float, 5, Eigen::Dynamic> EMatrix5Xf;
typedef EMatrix<float, Eigen::Dynamic, 5> EMatrixX5f;

typedef EMatrix<float, 6, 6> EMatrix6f;
typedef EMatrix<float, 6, Eigen::Dynamic> EMatrix6Xf;
typedef EMatrix<float, Eigen::Dynamic, 6> EMatrixX6f;

typedef EMatrix<float, Eigen::Dynamic, Eigen::Dynamic> EMatrixXf;

typedef EMatrix<int, 2, 2> EMatrix2i;
typedef EMatrix<int, 2, Eigen::Dynamic> EMatrix2Xi;
typedef EMatrix<int, Eigen::Dynamic, 2> EMatrixX2i;

typedef EMatrix<int, 3, 3> EMatrix3i;
typedef EMatrix<int, 3, Eigen::Dynamic> EMatrix3Xi;
typedef EMatrix<int, Eigen::Dynamic, 3> EMatrixX3i;

typedef EMatrix<int, 4, 4> EMatrix4i;
typedef EMatrix<int, 4, Eigen::Dynamic> EMatrix4Xi;
typedef EMatrix<int, Eigen::Dynamic, 4> EMatrixX4i;

typedef EMatrix<int, 5, 5> EMatrix5i;
typedef EMatrix<int, 5, Eigen::Dynamic> EMatrix5Xi;
typedef EMatrix<int, Eigen::Dynamic, 5> EMatrixX5i;

typedef EMatrix<int, 6, 6> EMatrix6i;
typedef EMatrix<int, 6, Eigen::Dynamic> EMatrix6Xi;
typedef EMatrix<int, Eigen::Dynamic, 6> EMatrixX6i;

typedef EMatrix<int, Eigen::Dynamic, Eigen::Dynamic> EMatrixXi;

typedef EMatrix<double, 1, 2> ERowVector2d;
typedef EMatrix<double, 1, 3> ERowVector3d;
typedef EMatrix<double, 1, 4> ERowVector4d;
typedef EMatrix<double, 1, 5> ERowVector5d;
typedef EMatrix<double, 1, 6> ERowVector6d;
typedef EMatrix<double, 1, Eigen::Dynamic> ERowVectorXd;

typedef EMatrix<float, 1, 2> ERowVector2f;
typedef EMatrix<float, 1, 3> ERowVector3f;
typedef EMatrix<float, 1, 4> ERowVector4f;
typedef EMatrix<float, 1, 5> ERowVector5f;
typedef EMatrix<float, 1, 6> ERowVector6f;
typedef EMatrix<float, 1, Eigen::Dynamic> ERowVectorXf;

typedef EMatrix<int, 1, 2> ERowVector2i;
typedef EMatrix<int, 1, 3> ERowVector3i;
typedef EMatrix<int, 1, 4> ERowVector4i;
typedef EMatrix<int, 1, 5> ERowVector5i;
typedef EMatrix<int, 1, 6> ERowVector6i;
typedef EMatrix<int, 1, Eigen::Dynamic> ERowVectorXi;

typedef EMatrix<double, 2, 1> EVector2d;
typedef EMatrix<double, 3, 1> EVector3d;
typedef EMatrix<double, 4, 1> EVector4d;
typedef EMatrix<double, 5, 1> EVector5d;
typedef EMatrix<double, 6, 1> EVector6d;
typedef EMatrix<double, Eigen::Dynamic, 1> EVectorXd;

typedef EMatrix<float, 2, 1> EVector2f;
typedef EMatrix<float, 3, 1> EVector3f;
typedef EMatrix<float, 4, 1> EVector4f;
typedef EMatrix<float, 5, 1> EVector5f;
typedef EMatrix<float, 6, 1> EVector6f;
typedef EMatrix<float, Eigen::Dynamic, 1> EVectorXf;

typedef EMatrix<int, 2, 1> EVector2i;
typedef EMatrix<int, 3, 1> EVector3i;
typedef EMatrix<int, 4, 1> EVector4i;
typedef EMatrix<int, 5, 1> EVector5i;
typedef EMatrix<int, 6, 1> EVector6i;
typedef EMatrix<int, Eigen::Dynamic, 1> EVectorXi;

#endif
