#ifndef UMUQ_HERMITEPOLYNOMIAL_H
#define UMUQ_HERMITEPOLYNOMIAL_H

#include "core/core.hpp"

#include <cstddef>

#include <vector>
#include <type_traits>
#include <utility>

namespace umuq
{

inline namespace polynomials
{

/*! \class HermitePolynomial
 * \ingroup Polynomials_Module
 *
 * \brief Hermite Polynomials
 *
 * \tparam RealType Floating-point data type
 *
 * \f$ H_n(x) \f$ is the physicist's Hermite polynomial of degree \f$ n \f$ and argument \f$ x. \f$
 *
 *
 * <table>
 * <caption id="multi_row">The first few physicists' Hermite polynomials are</caption>
 * <tr><th> n <th> \f$ ~~~~~H_n(x) \f$
 * <tr><td> 0 <td> \f$ ~~1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$
 * <tr><td> 1 <td> \f$ ~~2~x~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$
 * <tr><td> 2 <td> \f$ ~~4~x^2~-~~~~2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$
 * <tr><td> 3 <td> \f$ ~~8~x^3~-~~~~12~x~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$
 * <tr><td> 4 <td> \f$ ~~16~x^4~-~~~48~x^2~+~~~~12~~~~~~~~~~~~~~~~ \f$
 * <tr><td> 5 <td> \f$ ~~32~x^5~-~~~160~x^3~+~~~120~x~~~~~~~~~~~~~ \f$
 * <tr><td> 6 <td> \f$ ~~64~x^6~-~~~480~x^4~+~~~720~x^2~-~~~120~~~ \f$
 * <tr><td> 7 <td> \f$ ~~128~x^7~-~~1344~x^5~+~~3360~x^3~-~~1680~x \f$
 * <tr>
 * </table>
 *
 *
 * Reference:<br>
 * https://en.wikipedia.org/wiki/Hermite_polynomials
 *
 *
 * The results of this class is similar to the multivariate monomials with the degree of \b r in a space of \b d dimensions. \sa umuq::polynomials::polynomial
 *
 * A Hermite monomial in \f$ 1 \f$ variable \f$ x \f$ is simply any (non-negative integer) series of \f$ H_n(x) \f$:<br>
 *
 * \f$  H_0(x), H_1(x), H_2(x), H_3(x), \cdots, H_r(x) \f$<br>
 *
 * The highest exponent of \f$ x \f$ is termed the \b degree of the Hermite monomial.
 */
template <typename RealType>
class HermitePolynomial : public polynomialBase<RealType>
{
  public:
    /*!
     * \brief Construct a new Hermite Polynomial object
     *
     * \param dim              Dimension
     * \param PolynomialOrder  Polynomial order (the default order or degree of \b r in a space of dim dimensions is 2)
     */
    HermitePolynomial(int const dim, int const PolynomialOrder = 2);

    /*!
     * \brief Here, \f$\alpha=\f$ all of the Hermite monomials in a d dimensional space, with total degree \b r.
     *
     * For example: <br>
     * \verbatim
     *       d = 2
     *       r = 2
     *
     *       alpha[ 0],[ 1] = 0, 0 = H_0(x) H_0(y) = 1
     *       alpha[ 2],[ 3] = 1, 0 = H_1(x) H_0(y) = 2x
     *       alpha[ 4],[ 5] = 0, 1 = H_0(x) H_1(y) = 2y
     *       alpha[ 6],[ 7] = 2, 0 = H_2(x) H_0(y) = (4x^2-2)
     *       alpha[ 8],[ 9] = 1, 1 = H_1(x) H_1(y) = 4xy
     *       alpha[10],[11] = 0, 2 = H_0(x) H_2(y) = (4y^2-2)
     *
     *       monomialBasis_(d=2,r=2)   = {1,   H_1(x), H_1(y), H_2(x), H_1(x)H_1(y),  H_2(y)}
     *                           alpha = {0,0,   1,0,  0,1,      2,0,      1,1,       0,2}
     *
     *
     *       d = 3
     *       r = 2
     *
     *       alpha[ 0],[ 1],[ 2] = 0, 0, 0 = H_0(x) H_0(y) H_0(z) = 1
     *       alpha[ 3],[ 4],[ 5] = 1, 0, 0 = H_1(x) H_0(y) H_0(z) = 2x
     *       alpha[ 6],[ 7],[ 8] = 0, 1, 0 = H_0(x) H_1(y) H_0(z) = 2y
     *       alpha[ 9],[10],[11] = 0, 0, 1 = H_0(x) H_0(y) H_1(z) = 2z
     *       alpha[12],[13],[14] = 2, 0, 0 = H_2(x) H_0(y) H_0(z) = (4x^2-2)
     *       alpha[15],[16],[17] = 1, 1, 0 = H_1(x) H_1(y) H_0(z) = 4xy
     *       alpha[18],[19],[20] = 1, 0, 1 = H_1(x) H_0(y) H_1(z) = 4xz
     *       alpha[21],[22],[23] = 0, 2, 0 = H_0(x) H_2(y) H_0(z) = (4y^2-2)
     *       alpha[24],[25],[26] = 0, 1, 1 = H_0(x) H_1(y) H_1(z) = 4yz
     *       alpha[27],[28],[29] = 0, 0, 2 = H_0(x) H_0(y) H_2(z) = (4z^2-2)
     *
     * \endverbatim
     *
     * \returns A pointer to monomial sequence
     */
    int *monomialBasis();

    /*!
     * \brief Evaluates a monomial at a point x.
     *
     * \param  x       The coordinates of the evaluation points
     * \param  value   The (monomial value) array value of the monomial at point x
     *
     * \returns The size of the monomial array
     */
    int monomialValue(RealType const *x, RealType *&value);

    /*!
     * \brief Evaluates a monomial at a point x.
     *
     * \param  x       The coordinates of the evaluation points
     * \param  value   The (monomial value) array value of the monomial at point x
     *
     * \returns The size of the monomial array
     */
    int monomialValue(RealType const *x, std::vector<RealType> &value);

    /*!
     * \brief Computes the next Hermite polynomial of the degree n and argument x from the last two polynomial calculated.
     *
     * Computes the next Hermite polynomial of the degree n and argument x from the last two polynomial calculated.
     * Recurrence relation for Hermite P and Q polynomials.
     *
     * \param n     The degree of the last polynomial calculated.
     * \param x     The abscissa value.
     * \param Pn    The value of the polynomial evaluated at degree n.
     * \param Pnm1  The value of the polynomial evaluated at degree n-1.
     *
     * \returns RealType The computed Hermite Polynomial
     */
    inline RealType hermite_next(int const n, RealType const x, RealType const Pn, RealType const Pnm1);

    /*!
     * \brief Implement Hermite polynomials.
     *
     * This implementation contains minor change and adaptation to the [boost](https://www.boost.org)
     * source code made available under the following license: <br>
     *
     * \copyright
     * Boost Software License, Version 1.0. <br>
     * See the [LICENSE](http://www.boost.org/LICENSE_1_0.txt)
     *
     *
     * \param n  The degree of the Hermite polynomial \f$ H_n(x).\f$
     * \param x  The abscissa value.
     *
     * \returns RealType The Hermite polynomial of the degree n of x.
     */
    RealType hermite(int const n, RealType const x);

    /*!
     * \brief Implement Hermite polynomials.
     *
     * This implementation contains minor change and adaptation to the [boost](https://www.boost.org)
     * source code made available under the following license: <br>
     *
     * \copyright
     * Boost Software License, Version 1.0. <br>
     * See the [LICENSE](http://www.boost.org/LICENSE_1_0.txt)
     *
     *
     * \param n  The degree of the Hermite polynomial \f$ H_n(x).\f$
     * \param x  The abscissa value.
     *
     * \returns RealType* All the Hermite polynomials of the degrees \f$ 0, \cdots, n. \f$
     */
    RealType *hermite_array(int const n, RealType const x);

  private:
    /*!
     * \brief Delete a HermitePolynomial object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    HermitePolynomial(HermitePolynomial<RealType> const &) = delete;

    /*!
     * \brief Delete a HermitePolynomial object assignment
     *
     * Avoiding implicit copy assignment.
     */
    HermitePolynomial<RealType> &operator=(HermitePolynomial<RealType> const &) = delete;
};

template <typename RealType>
HermitePolynomial<RealType>::HermitePolynomial(int const dim, int const PolynomialOrder) : polynomialBase<RealType>(dim, PolynomialOrder)
{
    if (!std::is_floating_point<RealType>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }
}

template <typename RealType>
int *HermitePolynomial<RealType>::monomialBasis()
{
    if (this->alpha)
    {
        return this->alpha.get();
    }
    else
    {
        int const N = this->nDim * this->monomialSize;

        std::vector<int> x(this->nDim, 0);

        try
        {
            this->alpha.reset(new int[N]);
        }
        catch (...)
        {
            UMUQFAILRETURNNULL("Failed to allocate memory!");
        }

        int n(0);

        for (;;)
        {
            for (int j = this->nDim - 1; j >= 0; j--, n++)
            {
                this->alpha[n] = x[j];
            }

            if (x[0] == this->Order)
            {
                return this->alpha.get();
            }

            if (!this->graded_reverse_lexicographic_order(x.data()))
            {
                return nullptr;
            }
        }

        return this->alpha.get();
    }
}

template <typename RealType>
int HermitePolynomial<RealType>::monomialValue(RealType const *x, RealType *&value)
{
    if (!this->alpha)
    {
        //H ave to create monomial sequence
        int *tmp = this->monomialBasis();

        if (tmp == nullptr)
        {
            UMUQFAIL("Something went wrong in creating monomial sequence!");
        }
    }

    if (value == nullptr)
    {
        try
        {
            value = new RealType[this->monomialSize];
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }

    std::vector<RealType *> HermitePolynomialsValues(this->nDim);

    for (int j = 0; j < this->nDim; j++)
    {
        HermitePolynomialsValues[j] = this->hermite_array(this->Order, x[j]);
    }

    for (int i = 0, k = 0; i < this->monomialSize; i++)
    {
        RealType v = static_cast<RealType>(1);
        for (int j = 0; j < this->nDim; j++, k++)
        {
            int const l = this->alpha[k];
            v *= HermitePolynomialsValues[j][l];
        }
        value[i] = v;
    }

    for (int j = 0; j < this->nDim; j++)
    {
        delete[] HermitePolynomialsValues[j];
    }

    return this->monomialSize;
}

template <typename RealType>
int HermitePolynomial<RealType>::monomialValue(RealType const *x, std::vector<RealType> &value)
{
    if (!this->alpha)
    {
        //H ave to create monomial sequence
        int *tmp = this->monomialBasis();

        if (tmp == nullptr)
        {
            UMUQFAIL("Something went wrong in creating monomial sequence!");
        }
    }

    if (value.size() < static_cast<std::size_t>(this->monomialSize))
    {
        value.resize(this->monomialSize);
    }

    std::vector<RealType *> HermitePolynomialsValues(this->nDim);

    for (int j = 0; j < this->nDim; j++)
    {
        HermitePolynomialsValues[j] = this->hermite_array(this->Order, x[j]);
    }

    for (int i = 0, k = 0; i < this->monomialSize; i++)
    {
        RealType v = static_cast<RealType>(1);
        for (int j = 0; j < this->nDim; j++, k++)
        {
            int const l = this->alpha[k];
            v *= HermitePolynomialsValues[j][l];
        }
        value[i] = v;
    }

    for (int j = 0; j < this->nDim; j++)
    {
        delete[] HermitePolynomialsValues[j];
    }

    return this->monomialSize;
}

template <typename RealType>
inline RealType HermitePolynomial<RealType>::hermite_next(int const n, RealType const x, RealType const Pn, RealType const Pnm1)
{
    return (static_cast<RealType>(2) * x * Pn - static_cast<RealType>(2) * static_cast<RealType>(n) * Pnm1);
}

template <typename RealType>
RealType HermitePolynomial<RealType>::hermite(int const n, RealType const x)
{
    if (n == 0)
    {
        return static_cast<RealType>(1);
    }
    {
        RealType P0(1);
        RealType P1(static_cast<RealType>(2) * x);
        int i(1);
        while (i < n)
        {
            std::swap(P0, P1);
            P1 = this->hermite_next(i, x, P0, P1);
            ++i;
        }
        return P1;
    }
}

template <typename RealType>
RealType *HermitePolynomial<RealType>::hermite_array(int const n, RealType const x)
{
    RealType *results = nullptr;
    try
    {
        results = new RealType[n + 1];
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
    RealType P0 = static_cast<RealType>(1);
    results[0] = P0;
    if (n == 0)
    {
        return results;
    }
    {
        RealType P1 = static_cast<RealType>(2) * x;
        results[1] = P1;
        int i(1);
        while (i < n)
        {
            std::swap(P0, P1);
            P1 = this->hermite_next(i, x, P0, P1);
            ++i;
            results[i] = P1;
        }
        return results;
    }
}

} // namespace polynomials
} // namespace umuq

#endif // UMUQ_HERMITEPOLYNOMIAL
