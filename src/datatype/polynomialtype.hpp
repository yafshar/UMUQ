#ifndef UMUQ_POLYNOMIALTYPE_H
#define UMUQ_POLYNOMIALTYPE_H

namespace umuq
{

inline namespace polynomials
{

/*!
 * \ingroup Polynomials_Module
 * 
 * \brief Different Polynomials, currently available in %UMUQ
 * 
 */
enum class PolynomialTypes
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

#endif // UMUQ_POLYNOMIALTYPE
