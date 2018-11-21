#ifndef UMUQ_EIGENDATATYPE_H
#define UMUQ_EIGENDATATYPE_H

#include <Eigen/Eigen>

namespace umuq
{

/*! 
 * \defgroup LinearAlgebra_Module Linear algebra module
 * \ingroup Numerics_Module
 * 
 * This is the linear algebra module of %UMUQ providing all necessary classes for linear algebra: matrices, vectors, numerical solvers, and related algorithms.<br>
 * [Eigen](http://eigen.tuxfamily.org) (a C++ template library for linear algebra) forms the core mathematics library of %UMUQ, with all its linear algebra routines.
 * 
 * Reference:<br>
 * <a href="http://eigen.tuxfamily.org/"> Eigen C++ template library </a> 
 */

/*! 
 * \namespace umuq::linearalgebra
 * \ingroup LinearAlgebra_Module
 * 
 * \brief Namespace containing all the functions for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
 * 
 */
inline namespace linearalgebra
{

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience matrix data type. 
 * A rectangular \c Eigen::Matrix of T type with dynamic numbers of rows and dynamic numbers of columns. (dynamic_size_storage)
 * 
 * \tparam T Data type 
 * \tparam _Options  optional parameter, a combination of either 
 *                   - \b Eigen::RowMajor or \b Eigen::ColMajor, <br>
 *                     or one of either <br>
 *                   - \b Eigen::AutoAlign or \b Eigen::DontAlign. <br>
 *                   The former controls storage order, and defaults to column-major. The latter controls alignment, which is required
 *                   for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
 */
template <typename T, int _Options = Eigen::ColMajor>
using EMatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type and size of \f$ 2 \times 2\f$. (fixed_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix2 = Eigen::Matrix<T, 2, 2>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with 2 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix2X = Eigen::Matrix<T, 2, Eigen::Dynamic>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with dynamic number of rows and 2 columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrixX2 = Eigen::Matrix<T, Eigen::Dynamic, 2>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type and size of \f$ 3 \times 3\f$. (fixed_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix3 = Eigen::Matrix<T, 3, 3>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with 3 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix3X = Eigen::Matrix<T, 3, Eigen::Dynamic>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with dynamic number of rows and 3 columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrixX3 = Eigen::Matrix<T, Eigen::Dynamic, 3>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type and size of \f$ 4 \times 4\f$. (fixed_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix4 = Eigen::Matrix<T, 4, 4>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with 4 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix4X = Eigen::Matrix<T, 4, Eigen::Dynamic>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with dynamic number of rows and 4 columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrixX4 = Eigen::Matrix<T, Eigen::Dynamic, 4>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type and size of \f$ 5 \times 5\f$. (fixed_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix5 = Eigen::Matrix<T, 5, 5>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with 5 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix5X = Eigen::Matrix<T, 5, Eigen::Dynamic>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with dynamic number of rows and 5 columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrixX5 = Eigen::Matrix<T, Eigen::Dynamic, 5>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type and size of \f$ 6 \times 6\f$. (fixed_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix6 = Eigen::Matrix<T, 6, 6>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with 6 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrix6X = Eigen::Matrix<T, 6, Eigen::Dynamic>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of T type with dynamic number of rows and 6 columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
template <typename T>
using EMatrixX6 = Eigen::Matrix<T, Eigen::Dynamic, 6>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with dynamic numbers of rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 * 
 * \tparam T Data type
 */
using EMatrixXd = EMatrixX<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type and size of \f$ 2 \times 2\f$. (fixed_size_storage with column storage order.)
 */
using EMatrix2d = EMatrix2<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with 2 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 */
using EMatrix2Xd = EMatrix2X<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with dynamic number of rows and 2 columns. (dynamic_size_storage with column storage order.)
 */
using EMatrixX2d = EMatrixX2<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type and size of \f$ 3 \times 3\f$. (fixed_size_storage with column storage order.)
 */
using EMatrix3d = EMatrix3<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with 3 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 */
using EMatrix3Xd = EMatrix3X<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with dynamic number of rows and 3 columns. (dynamic_size_storage with column storage order.)
 */
using EMatrixX3d = EMatrixX3<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type and size of \f$ 4 \times 4\f$. (fixed_size_storage with column storage order.)
 */
using EMatrix4d = EMatrix4<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with 4 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 */
using EMatrix4Xd = EMatrix4X<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with dynamic number of rows and 4 columns. (dynamic_size_storage with column storage order.)
 */
using EMatrixX4d = EMatrixX4<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type and size of \f$ 5 \times 5\f$. (fixed_size_storage with column storage order.)
 */
using EMatrix5d = EMatrix5<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with 5 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 */
using EMatrix5Xd = EMatrix5X<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with dynamic number of rows and 5 columns. (dynamic_size_storage with column storage order.)
 */
using EMatrixX5d = EMatrixX5<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type and size of \f$ 6 \times 6\f$. (fixed_size_storage with column storage order.)
 */
using EMatrix6d = EMatrix6<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with 6 rows and dynamic numbers of columns. (dynamic_size_storage with column storage order.)
 */
using EMatrix6Xd = EMatrix6X<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A rectangular \c Eigen::Matrix of double type with dynamic number of rows and 6 columns. (dynamic_size_storage with column storage order.)
 */
using EMatrixX6d = EMatrixX6<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of T type with 1 rows and dynamic numbers of columns. (dynamic_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using ERowVectorX = Eigen::Matrix<T, 1, Eigen::Dynamic>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of T type with 1 rows and 2 columns. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using ERowVector2 = Eigen::Matrix<T, 1, 2>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of T type with 1 rows and 3 columns. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using ERowVector3 = Eigen::Matrix<T, 1, 3>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of T type with 1 rows and 4 columns. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using ERowVector4 = Eigen::Matrix<T, 1, 4>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of T type with 1 rows and 5 columns. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using ERowVector5 = Eigen::Matrix<T, 1, 5>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of T type with 1 rows and 6 columns. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using ERowVector6 = Eigen::Matrix<T, 1, 6>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief An Eigen row-vector of doubles data type
 * A rectangular \c Eigen::Matrix of double type with 1 rows and dynamic numbers of columns. (dynamic_size_storage)
 */
using ERowVectorXd = ERowVectorX<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of double type with 1 rows and 2 columns. (fixed_size_storage)
 */
using ERowVector2d = ERowVector2<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of double type with 1 rows and 3 columns. (fixed_size_storage)
 */
using ERowVector3d = ERowVector3<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of double type with 1 rows and 4 columns. (fixed_size_storage)
 */
using ERowVector4d = ERowVector4<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of double type with 1 rows and 5 columns. (fixed_size_storage)
 */
using ERowVector5d = ERowVector5<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience row-vector data type. 
 * A rectangular \c Eigen::Matrix of double type with 1 rows and 6 columns. (fixed_size_storage)
 */
using ERowVector6d = ERowVector6<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type. 
 * A rectangular \c Eigen::Matrix of T type with dynamic number of rows and 1 column. (dynamic_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using EVectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of T type with 2 rows and 1 column. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using EVector2 = Eigen::Matrix<T, 2, 1>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of T type with 3 rows and 1 column. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using EVector3 = Eigen::Matrix<T, 3, 1>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of T type with 4 rows and 1 column. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using EVector4 = Eigen::Matrix<T, 4, 1>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of T type with 5 rows and 1 column. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using EVector5 = Eigen::Matrix<T, 5, 1>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of T type with 6 rows and 1 column. (fixed_size_storage)
 * 
 * \tparam T Data type
 */
template <typename T>
using EVector6 = Eigen::Matrix<T, 6, 1>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type. 
 * A rectangular \c Eigen::Matrix of double type with dynamic number of rows and 1 column. (dynamic_size_storage)
 */
using EVectorXd = EVectorX<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of double type with 2 rows and 1 column. (fixed_size_storage)
 */
using EVector2d = EVector2<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of double type with 3 rows and 1 column. (fixed_size_storage)
 */
using EVector3d = EVector3<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of double type with 4 rows and 1 column. (fixed_size_storage)
 */
using EVector4d = EVector4<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of double type with 5 rows and 1 column. (fixed_size_storage)
 */
using EVector5d = EVector5<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief A convenience column-vector data type.
 * A rectangular \c Eigen::Matrix of double type with 6 rows and 1 column. (fixed_size_storage)
 */
using EVector6d = EVector6<double>;

/*!
 * \ingroup LinearAlgebra_Module
 * \brief Stores a set of parameters controlling the way matrices are printed.
 * 
 * - \b precision \c FullPrecision. <br>
 * - \b coeffSeparator string printed between two coefficients of the same row
 * - \b rowSeparator string printed between two rows
 */
Eigen::IOFormat eigenIOFormat(Eigen::FullPrecision);

} // namespace linearalgebra
} // namespace umuq

#endif // UMUQ_EIGENDATATYPE
