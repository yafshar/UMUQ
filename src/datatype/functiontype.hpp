#ifndef UMUQ_FUNCTIONTYPE_H
#define UMUQ_FUNCTIONTYPE_H

/*! \file functiontype.hpp
 * \brief Collection of Function types for convenience
 */

#include <functional>

namespace umuq
{

inline namespace multimin
{

/*!
 * \brief Function type that can be used in multidimensional minimization as \f$ f(x) \f$
 *
 */
template <typename DataType>
using F_MTYPE = std::function<DataType(DataType const *)>;

/*!
 * \brief differentiable Function type that can be used in multidimensional minimization as \f$ f(x, g) \f$
 *
 */
template <typename DataType>
using DF_MTYPE = std::function<bool(DataType const *, DataType *)>;

/*!
 * \brief Function & Derivative type that can be used in multidimensional minimization as \f$ fdf(x, f, g) \f$
 *
 */
template <typename DataType>
using FDF_MTYPE = std::function<bool(DataType const *, DataType *, DataType *)>;

} // namespace multimin

/*!
 * \brief Function type that can be used in fitFunction object
 *
 */
template <typename DataType>
using FITFUN_T = std::function<DataType(DataType const *, int const, DataType *, int const, int const *)>;

} // namespace umuq

#endif // UMUQ_FUNCTIONTYPE
