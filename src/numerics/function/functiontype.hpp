#ifndef UMUQ_FUNCTIONTYPE_H
#define UMUQ_FUNCTIONTYPE_H

/*! 
 * \brief Collection of Function types for convenience
 */

/*!
 * \brief Instances of std::function as f(x) 
 * 
 * \tparam T  IN/OUT data type
 */
template <typename T>
using FUN_x = std::function<T(T const)>;

/*!
 * \brief Instances of std::function as f(*x) 
 * 
 * \tparam T  IN/OUT data type
 */
template <typename T>
using FUN_x_p = std::function<T(T *)>;

/*!
 * \brief Instances of std::function as f(&x) 
 * 
 * \tparam T  OUT data type
 * \tparam V  IN data type
 */
template <typename T, class V>
using FUN_x_v = std::function<T(V const &)>;

/*!
 * \brief Instances of std::function as f(x,y) 
 * 
 * \tparam T IN/OUT data type
 * \tparam Y IN data type of the second variable 
 */
template <typename T, typename Y = T>
using FUN_xy = std::function<T(T const, Y const)>;

/*!
 * \brief Instances of std::function as f(*x,*y) 
 * 
 * \tparam T IN/OUT data type
 * \tparam T IN data type of the second variable
 */
template <typename T, typename Y = T>
using FUN_xy_p = std::function<T(T *, Y *)>;

/*!
 * \brief Function type that can be used in multidimensional minimization as \f$ f(x) \f$
 * 
 */
template <typename T>
using F_MTYPE = std::function<T(T const *)>;

/*!
 * \brief differentiable Function type that can be used in multidimensional minimization as \f$ f(x, g) \f$
 * 
 */
template <typename T>
using DF_MTYPE = std::function<bool(T const *, T *)>;

/*!
 * \brief Function & Derivative type that can be used in multidimensional minimization as \f$ fdf(x, f, g) \f$
 * 
 */
template <typename T>
using FDF_MTYPE = std::function<bool(T const *, T *, T *)>;

/*!
 * \brief Function type that can be used as \f$ f(x) \f$ where x is a const value
 * 
 */
template <typename T>
using F_LTYPE = std::function<T(T const)>;

/*!
 * \brief differentiable Function type that can be used as \f$ f(x, g) \f$, where x is a const value and g is the const gradient value
 */
template <typename T>
using DF_LTYPE = std::function<bool(T const, T *)>;

/*!
 * \brief Function & Derivative type that can be used as \f$ fdf(x, f, g) \f$  where x is a const value and f, and g are the const function & gradient values, respectively
 * 
 */
template <typename T>
using FDF_LTYPE = std::function<bool(T const, T *, T *)>;


#endif // UMUQ_FUNCTIONTYPE
