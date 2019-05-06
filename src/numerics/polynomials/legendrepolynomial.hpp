#ifndef UMUQ_LEGENDREPOLYNOMIAL_H
#define UMUQ_LEGENDREPOLYNOMIAL_H

namespace umuq
{

inline namespace polynomials
{

/*! \class LegendrePolynomial
 * \ingroup Polynomials_Module
 *
 * \brief Legendre Polynomials 
 * 
 * \tparam RealType Floating-point data type 
 * 
 * Legendre Polynomials are the polynomial solutions \f$ P_n(x) \f$ to the Legendre's differential equation<br>
 * 
 * \f$ \frac{d}{dx} \left[(1-x^2)\frac{dP_n(x)}{dx}\right]+n(n+1)P_n(x)=0, \f$ <br>
 * 
 * with integer parameter \f$ n \ge 0\f$ and with the convention:<br>
 * 
 * \f$ 
 * \begin{align}
 * \nonumber P_n(1) =& 1 \\
 * \nonumber P_n(-1) =& (-1)^n \\
 * \nonumber | P_n(x) | <=& 1, ~x \in [-1, 1]. 
 * \end{align}
 * \f$
 * 
 * The n zeroes of \f$ P_n(x) \f$ are the abscissas used for Gauss-Legendre quadrature of 
 * the integral of a function \f$ \int{f(x)dx} \f$ with weight function 1 over the interval \f$ [-1,1]. \f$
 * 
 * The Legendre polynomials are orthogonal under the inner product defined as integration from -1 to 1: <br>
 * 
 * \f$
 * \begin{align} 
 * \nonumber \int_{-1}^{1}{P_i(x)P_j(x)} dX =&~~0  &~\text{if}~ i \neq j \\
 * \nonumber                                =& \frac{2}{2i+1} &~\text{if}~ i = j. 
 * \end{align}
 * \f$
 * 
 * Except for \f$ P_0(x), \f$ the integral of \f$ P_i(x) \f$ from -1 to 1 is 0.
 * 
 * A function \f$ f(x) \f$ defined on \f$ [-1,1] \f$ may be approximated by the series <br>
 * 
 * \f$ C_0 P_0(x) + C_1 P_1(x) + \cdots + C_n P_n(x) \f$ 
 * 
 * where <br>
 * 
 * \f$ C_i = \frac{2i+1}{2} \int_{-1}^{1}{f(x)P_i(x)dx}. \f$
 * 
 * The formula is:<br>
 * 
 * \f$ P_n(x) = {(\frac{1}{2})}^n \sum_{m=0}^{n/2} C_n(m) C_{2n-2m}(n) x^{(n-2m)} \f$
 * 
 * Differential equation: <br>
 * 
 * \f$ (1-x \times x) {P_n(x)}'' - 2x \times \acute{P_n(x)} + n(n+1) = 0 \f$
 * 
 * The first few Legendre polynomials are:
 * 
 * <table>
 * <caption id="multi_row">Legendre polynomials</caption>
 * <tr><th> n <th> \f$ ~~~~~P_n(x) \f$        
 * <tr><td> 0 <td> \f$ (~1~)~~~ \f$ 
 * <tr><td> 1 <td> \f$ (~1~x~)~~~ \f$ 
 * <tr><td> 2 <td> \f$ (~3~x^2~-~~~~~1~)/2~ \f$ 
 * <tr><td> 3 <td> \f$ (~5~x^3~-~~~~~3~x~)/2~ \f$ 
 * <tr><td> 4 <td> \f$ (~35~x^4~-~~~~30~x^2~+~~~~3~)/8~ \f$ 
 * <tr><td> 5 <td> \f$ (~63~x^5~-~~~~70~x^3~+~~~~15~x~)/8~ \f$ 
 * <tr><td> 6 <td> \f$ (~231~x^6~-~~~315~x^4~+~~~105~x^2~-~~~~~5~)/16 \f$ 
 * <tr><td> 7 <td> \f$ (~429~x^7~-~~~693~x^5~+~~~315~x^3~-~~~~35~x)/16 \f$ 
 * <tr>
 * </table>
 * 
 * Recursion:<br>
 * 
 * \f$ 
 * \begin{align}
 * \nonumber P_0(x) =& 1 \\
 * \nonumber P_1(x) =& x \\
 * \nonumber P_n(x) =& \frac{(2n-1)~x}{n} P_{n-1}(x)-\frac{(n-1)}{n} P_{n-2}(x)
 * \end{align}
 * \f$
 * 
 * Reference:<br>
 * https://en.wikipedia.org/wiki/Legendre_polynomials
 *
 * The results of this class is similar to the multivariate monomials with the degree of \b r in a space of \b d dimensions. \sa umuq::polynomials::polynomial
 *
 * A Legendre monomial in \f$ 1 \f$ variable \f$ x \f$ is simply any (non-negative integer) series of \f$ P_n(x) \f$:<br>
 * 
 * \f$  P_0(x), P_1(x), P_2(x), P_3(x), \cdots, P_r(x) \f$<br>
 * 
 * The highest exponent of \f$ x \f$ is termed the \b degree of the Legendre monomial.
 */
template <typename RealType>
class LegendrePolynomial : public polynomialBase<RealType>
{
  public:
    /*!
     * \brief Construct a new Legendre Polynomial object
     * 
     * \param dim              Dimension
     * \param PolynomialOrder  Polynomial order (the default order or degree of \b r in a space of dim dimensions is 2)
     */
    LegendrePolynomial(int const dim, int const PolynomialOrder = 2);

    /*! 
     * \brief Here, \f$\alpha=\f$ all of the Legendre monomials in a d dimensional space, with total degree \b r.
     *   
     * For example: <br>
     * \verbatim
     *       d = 2
     *       r = 2
     *
     *       alpha[ 0],[ 1] = 0, 0 = P_0(x) P_0(y) = 1
     *       alpha[ 2],[ 3] = 1, 0 = P_1(x) P_0(y) = x
     *       alpha[ 4],[ 5] = 0, 1 = P_0(x) P_1(y) = y
     *       alpha[ 6],[ 7] = 2, 0 = P_2(x) P_0(y) = (3x^2-1)/2
     *       alpha[ 8],[ 9] = 1, 1 = P_1(x) P_1(y) = xy
     *       alpha[10],[11] = 0, 2 = P_0(x) P_2(y) = (3y^2-1)/2
     *
     *       monomialBasis_(d=2,r=2)   = {1,   P_1(x), P_1(y), P_2(x), P_1(x)P_1(y),  P_2(y)}
     *                           alpha = {0,0,   1,0,  0,1,      2,0,      1,1,       0,2}
     *
     * 
     *       d = 3
     *       r = 2
     *
     *       alpha[ 0],[ 1],[ 2] = 0, 0, 0 = P_0(x) P_0(y) P_0(z) = 1
     *       alpha[ 3],[ 4],[ 5] = 1, 0, 0 = P_1(x) P_0(y) P_0(z) = x
     *       alpha[ 6],[ 7],[ 8] = 0, 1, 0 = P_0(x) P_1(y) P_0(z) = y
     *       alpha[ 9],[10],[11] = 0, 0, 1 = P_0(x) P_0(y) P_1(z) = z
     *       alpha[12],[13],[14] = 2, 0, 0 = P_2(x) P_0(y) P_0(z) = (3x^2-1)/2
     *       alpha[15],[16],[17] = 1, 1, 0 = P_1(x) P_1(y) P_0(z) = xy
     *       alpha[18],[19],[20] = 1, 0, 1 = P_1(x) P_0(y) P_1(z) = xz
     *       alpha[21],[22],[23] = 0, 2, 0 = P_0(x) P_2(y) P_0(z) = (3y^2-1)/2
     *       alpha[24],[25],[26] = 0, 1, 1 = P_0(x) P_1(y) P_1(z) = yz
     *       alpha[27],[28],[29] = 0, 0, 2 = P_0(x) P_0(y) P_2(z) = (3z^2-1)/2
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
     * \brief Computes the next Legendre polynomial of the degree n and argument x from the last two polynomial calculated. 
     * 
     * Computes the next Legendre polynomial of the degree n and argument x from the last two polynomial calculated. 
     * Recurrence relation for legendre P and Q polynomials.
     * 
     * \param n     The degree of the last polynomial calculated. 
     * \param x     The abscissa value.
     * \param Pn    The value of the polynomial evaluated at degree n.
     * \param Pnm1  The value of the polynomial evaluated at degree n-1.
     * 
     * \returns RealType The computed Legendre Polynomial
     */
    inline RealType legendre_next(int const n, RealType const x, RealType const Pn, RealType const Pnm1);

    /*!
     * \brief Implement Legendre P and Q polynomials via recurrence.
     * 
     * This implementation contains minor change and adaptation to the [boost](https://www.boost.org)
     * source code made available under the following license: <br>
     * 
     * \copyright
     * Boost Software License, Version 1.0. <br>
     * See the [LICENSE](http://www.boost.org/LICENSE_1_0.txt)
     * 
     * 
     * \param n       The degree of the Legendre polynomial \f$ P_n(x).\f$ 
     * \param x       The abscissa value.
     * \param second  Request for the value of the Legendre polynomial that is the second solution to the Legendre differential equation.
     * 
     * \returns RealType The Legendre polynomial of the degree n \f$ P_n(x)~\text{or}~Q_n(x).\f$ 
     */
    RealType legendre(int const n, RealType const x, bool const second = false);

    /*!
     * \brief Implement Legendre P and Q polynomials via recurrence.
     * 
     * This implementation contains minor change and adaptation to the [boost](https://www.boost.org)
     * source code made available under the following license: <br>
     * 
     * \copyright
     * Boost Software License, Version 1.0. <br>
     * See the [LICENSE](http://www.boost.org/LICENSE_1_0.txt)
     * 
     * 
     * \param n       The degree of the Legendre polynomial \f$ P_n(x).\f$ 
     * \param x       The abscissa value.
     * \param second  Request for the value of the Legendre polynomial that is the second solution to the Legendre differential equation.
     * 
     * \returns RealType* All the Legendre polynomials of the degrees \f$ 0, \cdots, n. \f$
     */
    RealType *legendre_array(int const n, RealType const x, bool const second = false);

  private:
    /*!
     * \brief Delete a LegendrePolynomial object copy construction
     * 
     * Avoiding implicit generation of the copy constructor.
     */
    LegendrePolynomial(LegendrePolynomial<RealType> const &) = delete;

    /*!
     * \brief Delete a LegendrePolynomial object assignment
     * 
     * Avoiding implicit copy assignment.
     */
    LegendrePolynomial<RealType> &operator=(LegendrePolynomial<RealType> const &) = delete;
};

template <typename RealType>
LegendrePolynomial<RealType>::LegendrePolynomial(int const dim, int const PolynomialOrder) : polynomialBase<RealType>(dim, PolynomialOrder)
{
    if (!std::is_floating_point<RealType>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }
}

template <typename RealType>
int *LegendrePolynomial<RealType>::monomialBasis()
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
int LegendrePolynomial<RealType>::monomialValue(RealType const *x, RealType *&value)
{
    if (!this->alpha)
    {
        //Have to create monomial sequence
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

    std::vector<RealType *> legendrePolynomialsValues(this->nDim);

    for (int j = 0; j < this->nDim; j++)
    {
        legendrePolynomialsValues[j] = this->legendre_array(this->Order, x[j]);
    }

    for (int i = 0, k = 0; i < this->monomialSize; i++)
    {
        RealType v = static_cast<RealType>(1);
        for (int j = 0; j < this->nDim; j++, k++)
        {
            int const l = this->alpha[k];
            v *= legendrePolynomialsValues[j][l];
        }
        value[i] = v;
    }

    for (int j = 0; j < this->nDim; j++)
    {
        delete[] legendrePolynomialsValues[j];
    }

    return this->monomialSize;
}

template <typename RealType>
int LegendrePolynomial<RealType>::monomialValue(RealType const *x, std::vector<RealType> &value)
{
    if (!this->alpha)
    {
        //Have to create monomial sequence
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

    std::vector<RealType *> legendrePolynomialsValues(this->nDim);

    for (int j = 0; j < this->nDim; j++)
    {
        legendrePolynomialsValues[j] = this->legendre_array(this->Order, x[j]);
    }

    for (int i = 0, k = 0; i < this->monomialSize; i++)
    {
        RealType v = static_cast<RealType>(1);
        for (int j = 0; j < this->nDim; j++, k++)
        {
            int const l = this->alpha[k];
            v *= legendrePolynomialsValues[j][l];
        }
        value[i] = v;
    }

    for (int j = 0; j < this->nDim; j++)
    {
        delete[] legendrePolynomialsValues[j];
    }

    return this->monomialSize;
}

template <typename RealType>
inline RealType LegendrePolynomial<RealType>::legendre_next(int const n, RealType const x, RealType const Pn, RealType const Pnm1)
{
    return ((static_cast<RealType>(2) * static_cast<RealType>(n) + static_cast<RealType>(1)) * x * Pn - static_cast<RealType>(n) * Pnm1) / static_cast<RealType>(n + 1);
}

template <typename RealType>
RealType LegendrePolynomial<RealType>::legendre(int const n, RealType const x, bool const second)
{
    // Error handling:
    if ((x < -1) || (x > 1))
    {
        UMUQFAIL("The Legendre Polynomial is defined for  -1 <= x <= 1, but got x =", x);
    }

    RealType P0;
    RealType P1;
    if (second)
    {
        // A solution of the second kind (Q):
        P0 = (std::log1p(x) - std::log1p(-x)) / static_cast<RealType>(2);
        P1 = x * P0 - static_cast<RealType>(1);
    }
    else
    {
        // A solution of the first kind (P):
        P0 = static_cast<RealType>(1);
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
        P1 = this->legendre_next(i, x, P0, P1);
        ++i;
    }
    return P1;
}

template <typename RealType>
RealType *LegendrePolynomial<RealType>::legendre_array(int const n, RealType const x, bool const second)
{
    // Error handling:
    if ((x < -1) || (x > 1))
    {
        UMUQFAIL("The Legendre Polynomial is defined for  -1 <= x <= 1, but got x =", x);
    }

    RealType *results = nullptr;

    try
    {
        results = new RealType[n + 1];
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }

    RealType P0;
    RealType P1;
    if (second)
    {
        // A solution of the second kind (Q):
        P0 = (std::log1p(x) - std::log1p(-x)) / static_cast<RealType>(2);
        P1 = x * P0 - static_cast<RealType>(1);
    }
    else
    {
        // A solution of the first kind (P):
        P0 = static_cast<RealType>(1);
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
        P1 = this->legendre_next(i, x, P0, P1);
        ++i;
        results[i] = P1;
    }
    return results;
}

} // namespace polynomials
} // namespace umuq

#endif // UMUQ_LEGENDREPOLYNOMIAL

// template <class RealType>
// void uncheckedNonZeroLegendrePolynomialsCoefficientsExponents(unsigned int const n, RealType *Coefficients, int *Exponents)
// {
//     UMUQFAIL("The uncheckedNonZeroLegendrePolynomialsCoefficientsExponents is not implemented for this type!");
// }

// template <>
// void uncheckedNonZeroLegendrePolynomialsCoefficientsExponents<float>(unsigned int const n, float *Coefficients, int *Exponents)
// {
//     static float const FixedLegendrePolynomialsCoefficients[] =
//         {
//             1.0000000f,
//             1.0000000f,
//             -0.5000000f, 1.5000000f,
//             -1.5000000f, 2.5000000f,
//             0.37500000f, -3.7500000f, 4.3750000f,
//             1.87500000f, -8.7500000f, 7.8750000f,
//             -0.31250000f, 6.5625000f, -19.6875000f, 14.4375000f,
//             -2.1875000f, 19.6875000f, -43.3215000f, 26.8125000f,
//             0.27343750f, -9.8437500f, 54.14062500f, -93.8437500f, 50.2734375f,
//             2.46093750f, -36.0937500f, 140.7656250f, -201.093750f, 94.9609375f,
//             -0.24609375f, 13.53515625f, -117.3046875f, 351.9140625f, -427.32421875f, 180.42578125f};
//     static int const FixedLegendrePolynomialsExponents[] =
//         {
//             0,
//             1,
//             0, 2,
//             1, 3,
//             0, 2, 4,
//             1, 3, 5,
//             0, 2, 4, 6,
//             1, 3, 5, 7,
//             0, 2, 4, 6, 8,
//             1, 3, 5, 7, 9,
//             0, 2, 4, 6, 8, 10};
//     static std::size_t const startIndex[] = {0, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30};
//     std::size_t const S = startIndex[n];
//     std::size_t const E = S + (n + 2) / 2;
//     std::copy(FixedLegendrePolynomialsCoefficients + S, FixedLegendrePolynomialsCoefficients + E, Coefficients);
//     std::copy(FixedLegendrePolynomialsExponents + S, FixedLegendrePolynomialsExponents + E, Exponents);
//     return;
// }

// template <>
// void uncheckedNonZeroLegendrePolynomialsCoefficientsExponents<double>(unsigned int const n, double *Coefficients, int *Exponents)
// {
//     static double const FixedLegendrePolynomialsCoefficients[] =
//         {
//             1.0000000,
//             1.0000000,
//             -0.5000000, 1.5000000,
//             -1.5000000, 2.5000000,
//             0.37500000, -3.7500000, 4.3750000,
//             1.87500000, -8.7500000, 7.8750000,
//             -0.31250000, 6.5625000, -19.6875000, 14.4375000,
//             -2.1875000, 19.6875000, -43.3215000, 26.8125000,
//             0.27343750, -9.8437500, 54.14062500, -93.8437500, 50.2734375,
//             2.46093750, -36.0937500, 140.7656250, -201.093750, 94.9609375,
//             -0.24609375, 13.53515625, -117.3046875, 351.9140625, -427.32421875, 180.42578125};
//     static int const FixedLegendrePolynomialsExponents[] =
//         {
//             0,
//             1,
//             0, 2,
//             1, 3,
//             0, 2, 4,
//             1, 3, 5,
//             0, 2, 4, 6,
//             1, 3, 5, 7,
//             0, 2, 4, 6, 8,
//             1, 3, 5, 7, 9,
//             0, 2, 4, 6, 8, 10};
//     static std::size_t const startIndex[] = {0, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30};
//     std::size_t const S = startIndex[n];
//     std::size_t const E = S + (n + 2) / 2;
//     std::copy(FixedLegendrePolynomialsCoefficients + S, FixedLegendrePolynomialsCoefficients + E, Coefficients);
//     std::copy(FixedLegendrePolynomialsExponents + S, FixedLegendrePolynomialsExponents + E, Exponents);
//     return;
// }

// template <>
// void uncheckedNonZeroLegendrePolynomialsCoefficientsExponents<long double>(unsigned int const n, long double *Coefficients, int *Exponents)
// {
//     static long double const FixedLegendrePolynomialsCoefficients[] =
//         {
//             1.0000000l,
//             1.0000000l,
//             -0.5000000l, 1.5000000l,
//             -1.5000000l, 2.5000000l,
//             0.37500000l, -3.7500000l, 4.3750000l,
//             1.87500000l, -8.7500000l, 7.8750000l,
//             -0.31250000l, 6.5625000l, -19.6875000l, 14.4375000l,
//             -2.1875000l, 19.6875000l, -43.3215000l, 26.8125000l,
//             0.27343750l, -9.8437500l, 54.14062500l, -93.8437500l, 50.2734375l,
//             2.46093750l, -36.0937500l, 140.7656250l, -201.093750l, 94.9609375l,
//             -0.24609375l, 13.53515625l, -117.3046875l, 351.9140625l, -427.32421875l, 180.42578125l};
//     static int const FixedLegendrePolynomialsExponents[] =
//         {
//             0,
//             1,
//             0, 2,
//             1, 3,
//             0, 2, 4,
//             1, 3, 5,
//             0, 2, 4, 6,
//             1, 3, 5, 7,
//             0, 2, 4, 6, 8,
//             1, 3, 5, 7, 9,
//             0, 2, 4, 6, 8, 10};
//     static std::size_t const startIndex[] = {0, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30};
//     std::size_t const S = startIndex[n];
//     std::size_t const E = S + (n + 2) / 2;
//     std::copy(FixedLegendrePolynomialsCoefficients + S, FixedLegendrePolynomialsCoefficients + E, Coefficients);
//     std::copy(FixedLegendrePolynomialsExponents + S, FixedLegendrePolynomialsExponents + E, Exponents);
//     return;
// }