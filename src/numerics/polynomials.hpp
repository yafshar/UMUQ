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

/*!
 * \ingroup Polynomials_Module
 * 
 * \brief Different Polynomials, currently available in %UMUQ
 * 
 */
enum PolynomialTypes
{
	/*! \link umuq::polynomials::polynomial Polynomials. \endlink  */
	MONOMIAL = 1,
	/*! \link umuq::polynomials::LegendrePolynomial Legendre Polynomials. \endlink  */
	LEGENDRE = 2,
	/*! \link umuq::polynomials::HermitePolynomial Hermite Polynomials. \endlink */
	HERMITE = 3,
	/*! \link umuq::polynomials::ChebyshevPolynomial Chebyshev Polynomials. \endlink */
	CHEBYSHEV = 4
};

} // namespace polynomials
} // namespace umuq

#include "./polynomials/polynomialbase.hpp"
#include "./polynomials/polynomial.hpp"
#include "./polynomials/legendrepolynomial.hpp"
#include "./polynomials/hermitepolynomial.hpp"
#include "./polynomials/chebyshevpolynomial.hpp"

#endif // UMUQ_POLYNOMIALS
