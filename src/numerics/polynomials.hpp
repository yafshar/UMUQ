#ifndef UMUQ_POLYNOMIALS_H
#define UMUQ_POLYNOMIALS_H

namespace umuq
{

/*! \defgroup Polynomials_Module Polynomials module
 * \ingroup Numerics_Module
 *
 * This is the Polynomials module of %UMUQ providing all necessary classes
 * for creating different polynomials, required in the %UMUQ.
 */

/*! \namespace umuq::polynomials
 * \ingroup Numerics_Module
 *
 * \brief Namespace containing all the functions and classes for creating different polynomials.
 *
 * It includes all the functionalities for creating different polynomials.
 */
inline namespace polynomials
{
} // namespace polynomials
} // namespace umuq

#include "datatype/polynomialtype.hpp"
#include "polynomials/polynomialbase.hpp"
#include "polynomials/polynomial.hpp"
#include "polynomials/legendrepolynomial.hpp"
#include "polynomials/hermitepolynomial.hpp"
#include "polynomials/chebyshevpolynomial.hpp"

#endif // UMUQ_POLYNOMIALS
