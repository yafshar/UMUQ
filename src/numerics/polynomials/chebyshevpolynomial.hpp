#ifndef UMUQ_CHEBYSHEVPOLYNOMIAL_H
#define UMUQ_CHEBYSHEVPOLYNOMIAL_H

namespace umuq
{

inline namespace polynomials
{

/*! \class ChebyshevPolynomial
 * \ingroup Polynomials_Module
 *
 * \brief Chebyshev Polynomials 
 * 
 * \tparam RealType Floating-point data type 
 * 
 * Chebyshev polynomials, are a sequence of orthogonal polynomials. 
 * Chebyshev polynomials of the first kind are denoted \f$ T_n(x) \f$ and 
 * Chebyshev polynomials of the second kind are denoted \f$ U_n(x) \f$.
 * Chebyshev polynomials are important in approximation theory because the 
 * roots of the Chebyshev polynomials of the first kind, which are also called 
 * Chebyshev nodes, are used as nodes in polynomial interpolation. 
 * 
 * 
 * The first few Chebyshev polynomials of the first kind are:
 * 
 * <table>
 * <caption id="multi_row">Chebyshev polynomials of the first kind</caption>
 * <tr><th> n <th> \f$ ~~~~~T_n(x) \f$        
 * <tr><td> 0 <td> \f$ ~~1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 1 <td> \f$ ~~x~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 2 <td> \f$ ~~2~x^2~-~~~~1~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 3 <td> \f$ ~~4~x^3~-~~~~3~x~~~~~~~~~~~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 4 <td> \f$ ~~8~x^4~-~~~~8~x^2~+~~~~~1~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 5 <td> \f$ ~~16~x^5~-~~~20~x^3~+~~~~5~x~~~~~~~~~~~~ \f$ 
 * <tr><td> 6 <td> \f$ ~~32~x^6~-~~~48~x^4~+~~~~18~x^2~-~~~~1~~ \f$ 
 * <tr><td> 7 <td> \f$ ~~64~x^7~-~~~112~x^5~+~~~56~x^3~-~~~~7~x \f$ 
 * </table>
 * 
 * 
 * The first few Chebyshev polynomials of the second kind are:
 * 
 * <table>
 * <caption id="multi_row">Chebyshev polynomials of the second kind</caption>
 * <tr><th> n <th> \f$ ~~~~~U_n(x) \f$        
 * <tr><td> 0 <td> \f$ ~~1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 1 <td> \f$ ~~2x~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 2 <td> \f$ ~~4~x^2~-~~~~1~~~~~~~~~~~~~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 3 <td> \f$ ~~8~x^3~-~~~~4~x~~~~~~~~~~~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 4 <td> \f$ ~~16~x^4~-~~~12~x^2~+~~~~1~~~~~~~~~~~~~~ \f$ 
 * <tr><td> 5 <td> \f$ ~~32~x^5~-~~~32~x^3~+~~~~6~x~~~~~~~~~~~~ \f$ 
 * <tr><td> 6 <td> \f$ ~~64~x^6~-~~~80~x^4~+~~~~24~x^2~-~~~~1~~ \f$ 
 * <tr><td> 7 <td> \f$ ~~128~x^7~-~~192~x^5~+~~~80~x^3~-~~~~8~x \f$ 
 * </table>
 * 
 * 
 * Reference:<br>
 * https://en.wikipedia.org/wiki/Chebyshev_polynomials
 * 
 * 
 * The results of this class is similar to the multivariate monomials with the degree of \b r in a space of \b d dimensions. \sa umuq::polynomials::polynomial
 *
 * A Chebyshev monomial in \f$ 1 \f$ variable \f$ x \f$ is simply any (non-negative integer) series of \f$ T_n(x) \f$:<br>
 * 
 * \f$  T_0(x), T_1(x), T_2(x), T_3(x), \cdots, T_r(x) \f$<br>
 * 
 * The highest exponent of \f$ x \f$ is termed the \b degree of the Chebyshev monomial.
 */
template <typename RealType>
class ChebyshevPolynomial : public polynomialBase<RealType>
{
  public:
    /*!
     * \brief Construct a new Chebyshev Polynomial object
     * 
     * \param dim              Dimension
     * \param PolynomialOrder  Polynomial order (the default order or degree of \b r in a space of dim dimensions is 2)
     */
    ChebyshevPolynomial(int const dim, int const PolynomialOrder = 2);

    /*! 
     * \brief Here, \f$\alpha=\f$ all of the Chebyshev monomials in a d dimensional space, with total degree \b r.
     *   
     * For example: <br>
     * \verbatim
     *       d = 2
     *       r = 2
     *
     *       alpha[ 0],[ 1] = 0, 0 = T_0(x) T_0(y) = 1
     *       alpha[ 2],[ 3] = 1, 0 = T_1(x) T_0(y) = x
     *       alpha[ 4],[ 5] = 0, 1 = T_0(x) T_1(y) = y
     *       alpha[ 6],[ 7] = 2, 0 = T_2(x) T_0(y) = (2x^2-1)
     *       alpha[ 8],[ 9] = 1, 1 = T_1(x) T_1(y) = xy
     *       alpha[10],[11] = 0, 2 = T_0(x) T_2(y) = (2y^2-1)
     *
     *       monomialBasis_(d=2,r=2)   = {1,   T_1(x), T_1(y), T_2(x), T_1(x)T_1(y),  T_2(y)}
     *                           alpha = {0,0,  1,0,    0,1,     2,0,      1,1,       0,2}
     *
     * 
     *       d = 3
     *       r = 2
     *
     *       alpha[ 0],[ 1],[ 2] = 0, 0, 0 = T_0(x) T_0(y) T_0(z) = 1
     *       alpha[ 3],[ 4],[ 5] = 1, 0, 0 = T_1(x) T_0(y) T_0(z) = x
     *       alpha[ 6],[ 7],[ 8] = 0, 1, 0 = T_0(x) T_1(y) T_0(z) = y
     *       alpha[ 9],[10],[11] = 0, 0, 1 = T_0(x) T_0(y) T_1(z) = z
     *       alpha[12],[13],[14] = 2, 0, 0 = T_2(x) T_0(y) T_0(z) = (2x^2-1)
     *       alpha[15],[16],[17] = 1, 1, 0 = T_1(x) T_1(y) T_0(z) = xy
     *       alpha[18],[19],[20] = 1, 0, 1 = T_1(x) T_0(y) T_1(z) = xz
     *       alpha[21],[22],[23] = 0, 2, 0 = T_0(x) T_2(y) T_0(z) = (2y^2-1)
     *       alpha[24],[25],[26] = 0, 1, 1 = T_0(x) T_1(y) T_1(z) = yz
     *       alpha[27],[28],[29] = 0, 0, 2 = T_0(x) T_0(y) T_2(z) = (2z^2-1)
     *
     * \endverbatim
     *
     * \returns A pointer to monomial sequence
     */
    int *monomialBasis();

    /*! 
     * \brief Evaluates a monomial at a point x.
     * 
     * \param  x      The coordinates of the evaluation points
     * \param  value  The (monomial value) array value of the monomial at point x
     * 
     * \returns int The size of the monomial array
     */
    int monomialValue(RealType const *x, RealType *&value);

    /*! 
     * \brief Evaluates a monomial at a point x.
     * 
     * \param  x      The coordinates of the evaluation points
     * \param  value  The (monomial value) array value of the monomial at point x
     * 
     * \returns int The size of the monomial array
     */
    int monomialValue(RealType const *x, std::vector<RealType> &value);

    /*!
     * \brief Computes the next Chebyshev polynomial of the degree n and argument x from the last two polynomial calculated. 
     * 
     * Computes the next Chebyshev polynomial of the degree n and argument x from the last two polynomial calculated. 
     * Recurrence relation for Chebyshev RealType and U polynomials.
     * 
     * \param x     The abscissa value.
     * \param Pn    The value of the polynomial evaluated at degree n.
     * \param Pnm1  The value of the polynomial evaluated at degree n-1.
     * 
     * \returns RealType The computed Chebyshev Polynomial 
     */
    inline RealType chebyshev_next(RealType const x, RealType const Pn, RealType const Pnm1);

    /*!
     * \brief Implement Chebyshev polynomials.
     * 
     * This implementation contains minor change and adaptation to the [boost](https://www.boost.org)
     * source code made available under the following license: <br>
     * 
     * \copyright
     * Boost Software License, Version 1.0. <br>
     * See the [LICENSE](http://www.boost.org/LICENSE_1_0.txt)
     * 
     * 
     * \param n  The degree of the Chebyshev polynomial \f$ T_n(x)~\text{or}~U_n(x).\f$ 
     * \param x  The abscissa value.
     * 
     * \returns RealType The Chebyshev polynomial of the degree n of x.
     */
    RealType chebyshev(int const n, RealType const x, bool const second = false);

    /*!
     * \brief Implement Chebyshev polynomials.
     * 
     * This implementation contains minor change and adaptation to the [boost](https://www.boost.org)
     * source code made available under the following license: <br>
     * 
     * \copyright
     * Boost Software License, Version 1.0. <br>
     * See the [LICENSE](http://www.boost.org/LICENSE_1_0.txt)
     * 
     * 
     * \param n  The degree of the Chebyshev polynomial \f$ T_n(x).\f$ 
     * \param x  The abscissa value.
     * 
     * \returns RealType* All the Chebyshev polynomials of the degrees \f$ 0, \cdots, n. \f$
     */
    RealType *chebyshev_array(int const n, RealType const x, bool const second = false);

  private:
    /*!
     * \brief Delete a ChebyshevPolynomial object copy construction
     * 
     * Avoiding implicit generation of the copy constructor.
     */
    ChebyshevPolynomial(ChebyshevPolynomial<RealType> const &) = delete;

    /*!
     * \brief Delete a ChebyshevPolynomial object assignment
     * 
     * Avoiding implicit copy assignment.
     */
    ChebyshevPolynomial<RealType> &operator=(ChebyshevPolynomial<RealType> const &) = delete;
};

template <typename RealType>
ChebyshevPolynomial<RealType>::ChebyshevPolynomial(int const dim, int const PolynomialOrder) : polynomialBase<RealType>(dim, PolynomialOrder)
{
    if (!std::is_floating_point<RealType>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }
}

template <typename RealType>
int *ChebyshevPolynomial<RealType>::monomialBasis()
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
int ChebyshevPolynomial<RealType>::monomialValue(RealType const *x, RealType *&value)
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

    std::vector<RealType *> ChebyshevPolynomialsValues(this->nDim);

    for (int j = 0; j < this->nDim; j++)
    {
        ChebyshevPolynomialsValues[j] = this->chebyshev_array(this->Order, x[j]);
    }

    for (int i = 0, k = 0; i < this->monomialSize; i++)
    {
        RealType v = static_cast<RealType>(1);
        for (int j = 0; j < this->nDim; j++, k++)
        {
            int const l = this->alpha[k];
            v *= ChebyshevPolynomialsValues[j][l];
        }
        value[i] = v;
    }

    for (int j = 0; j < this->nDim; j++)
    {
        delete[] ChebyshevPolynomialsValues[j];
    }

    return this->monomialSize;
}

template <typename RealType>
int ChebyshevPolynomial<RealType>::monomialValue(RealType const *x, std::vector<RealType> &value)
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

    std::vector<RealType *> ChebyshevPolynomialsValues(this->nDim);

    for (int j = 0; j < this->nDim; j++)
    {
        ChebyshevPolynomialsValues[j] = this->chebyshev_array(this->Order, x[j]);
    }

    for (int i = 0, k = 0; i < this->monomialSize; i++)
    {
        RealType v = static_cast<RealType>(1);
        for (int j = 0; j < this->nDim; j++, k++)
        {
            int const l = this->alpha[k];
            v *= ChebyshevPolynomialsValues[j][l];
        }
        value[i] = v;
    }

    for (int j = 0; j < this->nDim; j++)
    {
        delete[] ChebyshevPolynomialsValues[j];
    }

    return this->monomialSize;
}

template <typename RealType>
inline RealType ChebyshevPolynomial<RealType>::chebyshev_next(RealType const x, RealType const Pn, RealType const Pnm1)
{
    return (static_cast<RealType>(2) * x * Pn - Pnm1);
}

template <typename RealType>
RealType ChebyshevPolynomial<RealType>::chebyshev(int const n, RealType const x, bool const second)
{
    RealType P0(1);
    RealType P1;
    if (second)
    {
        if (x > 1 || x < -1)
        {
            RealType const tmp = std::sqrt(x * x - static_cast<RealType>(1));
            return (std::pow(x + tmp, n + 1) - std::pow(x - tmp, n + 1)) / (static_cast<RealType>(2) * tmp);
        }
        P1 = static_cast<RealType>(2) * x;
    }
    else
    {
        if (x > 1)
        {
            return std::cosh(static_cast<RealType>(n) * std::acosh(x));
        }
        if (x < -1)
        {
            if (n & 1)
            {
                return -std::cosh(static_cast<RealType>(n) * std::acosh(-x));
            }
            else
            {
                return std::cosh(static_cast<RealType>(n) * acosh(-x));
            }
        }
        P1 = x;
    }
    if (n == 0)
    {
        return P0;
    }

    int i(1);
    while (i < n)
    {
        std::swap(P0, P1);
        P1 = this->chebyshev_next(x, P0, P1);
        ++i;
    }
    return P1;
}

template <typename RealType>
RealType *ChebyshevPolynomial<RealType>::chebyshev_array(int const n, RealType const x, bool const second)
{
    RealType *results = nullptr;

    try
    {
        results = new RealType[n + 1];
    }
    catch (...)
    {
        UMUQFAILRETURNNULL("Failed to allocate memory!");
    }

    RealType P0(1);
    RealType P1;
    if (second)
    {
        if (x > 1 || x < -1)
        {
            RealType const tmp = std::sqrt(x * x - static_cast<RealType>(1));
            int j(0);
            std::for_each(results, results + n + 1, [&](RealType &r_i) { j++; r_i = (std::pow(x + tmp, j) - std::pow(x - tmp, j)) / (static_cast<RealType>(2) * tmp); });
            return results;
        }
        P1 = static_cast<RealType>(2) * x;
    }
    else
    {
        if (x > 1)
        {
            int j(0);
            std::for_each(results, results + n + 1, [&](RealType &r_i) {r_i = std::cosh(static_cast<RealType>(j) * std::acosh(x)); j++; });
            return results;
        }
        if (x < -1)
        {
            for (int j = 0; j < n + 1; j++)
            {
                if (j & 1)
                {
                    results[j] = -std::cosh(static_cast<RealType>(j) * std::acosh(-x));
                }
                else
                {
                    results[j] = std::cosh(static_cast<RealType>(j) * acosh(-x));
                }
            }
            return results;
        }
        P1 = x;
    }

    results[0] = P0;
    if (n == 0)
    {
        return results;
    }

    results[1] = P1;

    int i(1);
    while (i < n)
    {
        std::swap(P0, P1);
        P1 = this->chebyshev_next(x, P0, P1);
        ++i;
        results[i] = P1;
    }
    return results;
}

} // namespace polynomials
} // namespace umuq

#endif // UMUQ_CHEBYSHEVPOLYNOMIAL
