#ifndef UMUQ_EIGENDATATYPE_H
#define UMUQ_EIGENDATATYPE_H

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
using EMatrix2 = Eigen::Matrix<T, 2, 2>;                 // fixed_size_storage
template <typename T>                                 
using EMatrix2X = Eigen::Matrix<T, 2, Eigen::Dynamic>;   // dynamic_size_storage
template <typename T>
using EMatrixX2 = Eigen::Matrix<T, Eigen::Dynamic, 2>;   // dynamic_size_storage

template <typename T>
using EMatrix3 = Eigen::Matrix<T, 3, 3>;                 // fixed_size_storage
template <typename T>
using EMatrix3X = Eigen::Matrix<T, 3, Eigen::Dynamic>;   // dynamic_size_storage
template <typename T>
using EMatrixX3 = Eigen::Matrix<T, Eigen::Dynamic, 3>;   // dynamic_size_storage

template <typename T>
using EMatrix4 = Eigen::Matrix<T, 4, 4>;                 // fixed_size_storage
template <typename T>
using EMatrix4X = Eigen::Matrix<T, 4, Eigen::Dynamic>;   // dynamic_size_storage
template <typename T>
using EMatrixX4 = Eigen::Matrix<T, Eigen::Dynamic, 4>;   // dynamic_size_storage

template <typename T>
using EMatrix5 = Eigen::Matrix<T, 5, 5>;                 // fixed_size_storage
template <typename T>
using EMatrix5X = Eigen::Matrix<T, 5, Eigen::Dynamic>;   // dynamic_size_storage
template <typename T>
using EMatrixX5 = Eigen::Matrix<T, Eigen::Dynamic, 5>;   // dynamic_size_storage

template <typename T>
using EMatrix6 = Eigen::Matrix<T, 6, 6>;                 // fixed_size_storage
template <typename T>
using EMatrix6X = Eigen::Matrix<T, 6, Eigen::Dynamic>;   // dynamic_size_storage
template <typename T>
using EMatrixX6 = Eigen::Matrix<T, Eigen::Dynamic, 6>;   // dynamic_size_storage

/*!
 * \brief Eigen matrix of doubles data type
 * 
 */
using EMatrix2d = EMatrix2<double>;
using EMatrix2Xd = EMatrix2X<double>;
using EMatrixX2d = EMatrixX2<double>;
using EMatrix3d = EMatrix3<double>;
using EMatrix3Xd = EMatrix3X<double>;
using EMatrixX3d = EMatrixX3<double>;
using EMatrix4d = EMatrix4<double>;
using EMatrix4Xd = EMatrix4X<double>;
using EMatrixX4d = EMatrixX4<double>;
using EMatrix5d = EMatrix5<double>;
using EMatrix5Xd = EMatrix5X<double>;
using EMatrixX5d = EMatrixX5<double>;
using EMatrix6d = EMatrix6<double>;
using EMatrix6Xd = EMatrix6X<double>;
using EMatrixX6d = EMatrixX6<double>;

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

#endif // UMUQ_EIGENDATATYPE_H
