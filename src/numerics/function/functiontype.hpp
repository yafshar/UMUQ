#ifndef UMUQ_FUNCTIONTYPE_H
#define UMUQ_FUNCTIONTYPE_H

/*! 
 * \brief Collection of Function types for convenience
 */

namespace umuq
{
inline namespace multimin
{
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

} // namespace multimin
} // namespace umuq

/*!
 * \brief Function type that can be used in fitFunction object
 * 
 */
template <typename T>
using FITFUN_T = std::function<T(T const *, int const, T *, int const, int const *)>;

#endif // UMUQ_FUNCTIONTYPE
