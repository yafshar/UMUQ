#ifndef UMUQ_POLYNOMIAL_H
#define UMUQ_POLYNOMIAL_H

namespace umuq
{

/*! \class LegendrePolynomial
 * \ingroup Numerics_Module
 *
 * \brief Legendre Polynomials 
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
 * <table>
 * <caption id="multi_row">The first few Legendre polynomials</caption>
 * <tr><th> n <th> \f$ ~~~~~P_n(x) \f$        
 * <tr><td> 0 <td> \f$ (~~~~~1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~) \f$ 
 * <tr><td> 1 <td> \f$ (~~~~~1~x~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~) \f$ 
 * <tr><td> 2 <td> \f$ (~~~~3~x^2~-~~~~~~~1~~~~~~~~~~~~~~~~~~~~~~~~)/2 \f$ 
 * <tr><td> 3 <td> \f$ (~~~~5~x^3~-~~~~~3~x~~~~~~~~~~~~~~~~~~~~~~~~)/2 \f$ 
 * <tr><td> 4 <td> \f$ (~~~35~x^4~-~~~~30~x^2~+~~~~~3~~~~~~~~~~~~~~)/8 \f$ 
 * <tr><td> 5 <td> \f$ (~~~63~x^5~-~~~~70~x^3~+~~~~15~x~~~~~~~~~~~~)/8 \f$ 
 * <tr><td> 6 <td> \f$ (~~231~x^6~-~~~315~x^4~+~~~105~x^2~-~~~~~5~~)/16 \f$ 
 * <tr><td> 7 <td> \f$ (~~429~x^7~-~~~693~x^5~+~~~315~x^3~-~~~~35~x)/16 \f$ 
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
 * \Refernce:<br>
 * https://en.wikipedia.org/wiki/Legendre_polynomials
 * 
 * 
 * The results of this class is similar to the multivariate monomials with the degree of \b r in a space of \b d dimensions. \sa umuq::polynomial
 *
 * A Legendre monomial in \f$ 1 \f$ variable \f$ x \f$ is simply any (non-negative integer) series of \f$ P_n(x) \f$:<br>
 * \f$  P_0(x), P_1(x), P_2(x), P_3(x), \cdots, P_r(x) \f$<br>
 * The highest exponent of \f$ x \f$ is termed the \b degree of the Legendre monomial.
 */
template <typename T>
class LegendrePolynomial
{
  public:
    /*! 
     * \brief constructor
     * 
     * \param dim  Dimension
     * \param ord  Polynomial order (the default order or degree of r in a space of dim dimensions is 2)
     */
    LegendrePolynomial(int const dim, int const ord = 2);

    /*! 
     * \brief reset
     * 
     * Reset the values to the new ones
     * 
     * \param dim new Dimension
     * \param ord new Order (the default order or degree of r in a space of dm dimensions is 2)
     */
    void reset(int const dim, int const ord = 2);

    /*! 
     * \brief Computes the [binomial coefficient](https://en.wikipedia.org/wiki/Binomial_coefficient) \f$ C(n, k) \f$.
     *
     * -# A binomial coefficient \f$ C(n, k) \f$ can be defined as the coefficient of \f$ x ^ k \f$ in the expansion of \f$ (1 + x) ^ n. \f$ <br>
     * -# A binomial coefficient \f$ C(n, k) \f$ also gives the number of ways, disregarding order, that k objects can be 
     * chosen from among n objects; <br>
     * More formally, the number of k-element subsets (or k-combinations) of an n-element set.
     * 
     * The formula used is: <br>
     * \f$ C(n, k) = \frac{n!}{ n! * (n-k)! } \f$ 
     * 
     * \param n  Input parameter
     * \param k  Input parameter
     * 
     * \returns The binomial coefficient \f$ C(n, k) \f$
     */
    int binomialCoefficient(int const n, int const k);

    /*! 
     * \brief Here, \f$\alpha=\f$ all of the Legendre monomials in a d dimensional space, with total degree r.
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
    int monomialValue(T const *x, T *&value);

    /*!
     * \brief Get the monomial size
     * 
     * \return Monomial size
     */
    inline int monomialsize() const;

    /*!
     * \brief get the dimension
     * 
     * \return Dimension
     */
    inline int dim() const;

    /*!
     * \brief Polynomial order 
     * 
     * \return Polynomial order
     */
    inline int order() const;

  private:
    /*! 
     * \brief Use a reverse lexicographic order for next monomial, degrees between 0 and r
     *  all monomials in a d dimensional space, with order of accuracy r.
     *
     * \param x  Current monomial on input and next monomial on the output (last value in the sequence is r).
     */
    bool graded_reverse_lexicographic_order(int *x);

  public:
    /*!
     * \brief First coefficient (\f$ \frac{(2n+1)~x}{n+1} \f$) in the Legendre recursion formula.
     * 
     * Legendre recursion formula:<br>
     * 
     * \f$ P_{n+1}(x) =& \frac{(2n+1)~x}{n+1} P_{n}(x)-\frac{(n)}{n+1} P_{n-1}(x). \f$
     *  
     * \param n  The degree of the Legendre polynomial \f$ P_n(x).\f$ 
     * \param x  The abscissa value.
     * 
     * \returns constexpr T Coefficient value \f$ .
     */
    inline constexpr T Coefficient1(int const n, T const x);

    /*!
     * \brief Second coefficient (\f$ \frac{(n)}{n+1} \f$) in the Legendre recursion formula.
     * 
     * Legendre recursion formula:<br>
     * 
     * \f$ P_{n+1}(x) =& \frac{(2n+1)~x}{n+1} P_{n}(x)-\frac{(n)}{n+1} P_{n-1}(x). \f$
     *  
     * \param n  The degree of the Legendre polynomial \f$ P_n(x).\f$ 
     * \param x  The abscissa value.
     * 
     * \returns constexpr T Coefficient value \f$ .
     */
    inline constexpr T Coefficient2(int const n, T const x);

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
     * \returns T The computed Legendre Polynomial
     */
    inline T legendre_next(int const n, T const x, T const Pn, T const Pnm1);

    /*!
     * \brief Implement Legendre P and Q polynomials via recurrence.
     * 
     * This implementation contains minor change and adaptation to the [boost](https://www.boost.org)
     * source code made available under the following license: <br>
     * 
     * \copyright
     * \verbatim
     * Boost Software License, Version 1.0. (See accompanying file
     * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
     * \endverbatim
     * 
     * 
     * \param n       The degree of the Legendre polynomial \f$ P_n(x).\f$ 
     * \param x       The abscissa value.
     * \param second  Request for the value of the Legendre polynomial that is the second solution to the Legendre differential equation.
     * 
     * \returns T The Legendre polynomial of the degree n \f$ P_n(x)~\text{or}~Q_n(x).\f$ 
     */
    T legendre_p(int const n, T const x, bool const second = false);

    /*!
     * \brief Implement Legendre P and Q polynomials via recurrence.
     * 
     * This implementation contains minor change and adaptation to the [boost](https://www.boost.org)
     * source code made available under the following license: <br>
     * 
     * \copyright
     * \verbatim
     * Boost Software License, Version 1.0. (See accompanying file
     * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
     * \endverbatim
     * 
     * 
     * \param n       The degree of the Legendre polynomial \f$ P_n(x).\f$ 
     * \param x       The abscissa value.
     * \param second  Request for the value of the Legendre polynomial that is the second solution to the Legendre differential equation.
     * 
     * \returns T* All the Legendre polynomials of the degrees \f$ 0, \cdots, n. \f$
     */
    T *legendre_p_array(int const n, T const x, bool const second = false);

  private:
    /*!
     * \brief Delete a LegendrePolynomial object copy construction
     * 
     * Make it noncopyable.
     */
    LegendrePolynomial(LegendrePolynomial<T> const &) = delete;

    /*!
     * \brief Delete a LegendrePolynomial object assignment
     * 
     * Make it nonassignable
     * 
     * \returns LegendrePolynomial<T>& 
     */
    LegendrePolynomial<T> &operator=(LegendrePolynomial<T> const &) = delete;

  private:
    //! Dimension
    int nDim;

    //! Order of accuracy
    int Order;

    //! The size of the monomial array
    int monomialSize;

    //! Array of monomial sequence
    std::unique_ptr<int[]> alpha;
};

template <typename T>
LegendrePolynomial<T>::LegendrePolynomial(int const dim, int const ord) : nDim(dim), Order(ord)
{
    if (nDim <= 0)
    {
        UMUQFAIL("Can not have dimension = ", nDim, " <= 0!");
    }

    if (Order < 0)
    {
        UMUQFAIL("Maximum accuracy order ", Order, " < 0!");
    }

    monomialSize = binomialCoefficient(nDim + Order, Order);
    if (monomialSize == 0)
    {
        UMUQFAIL("Monomial size of zero degree is requested!");
    }
}

template <typename T>
void LegendrePolynomial<T>::reset(int const dim, int const ord)
{
    nDim = dim;
    if (nDim <= 0)
    {
        UMUQFAIL("Can not have dimension ", nDim, " <= 0!");
    }

    Order = ord;
    if (Order < 0)
    {
        UMUQFAIL("Maximum accuracy order ", Order, " < 0!");
    }

    monomialSize = binomialCoefficient(nDim + Order, Order);
    if (monomialSize == 0)
    {
        UMUQFAIL("The requested monomial size of zero is wrong!");
    }

    alpha.reset(nullptr);
}

template <typename T>
int LegendrePolynomial<T>::binomialCoefficient(int const n, int const k)
{
    if ((k < 0) || (n < 0))
    {
        UMUQFAIL("Fatal error! k=", k, " or n=", n, " < 0!");
    }
    if (k < n)
    {
        if (k == 0)
        {
            return 1;
        }
        if ((k == 1) || (k == n - 1))
        {
            return n;
        }

        int mn = std::min(k, n - k);
        int mx = std::max(k, n - k);
        int value = mx + 1;
        for (int i = 2; i <= mn; i++)
        {
            value = (value * (mx + i)) / i;
        }

        return value;
    }
    else if (k == n)
    {
        return 1;
    }

    UMUQWARNING("The binomial coefficient is undefined for k=", k, " > n=", n, " !");
    return 0;
}

template <typename T>
inline constexpr T LegendrePolynomial<T>::Coefficient1(int const n, T const x)
{
    return (static_cast<T>(2) * static_cast<T>(n) + static_cast<T>(1)) * x / static_cast<T>(n + 1);
}

template <typename T>
inline constexpr T LegendrePolynomial<T>::Coefficient2(int const n, T const x)
{
    return static_cast<T>(n) / static_cast<T>(n + 1);
}

template <typename T>
inline T LegendrePolynomial<T>::legendre_next(int const n, T const x, T const Pn, T const Pnm1)
{
    return ((static_cast<T>(2) * static_cast<T>(n) + static_cast<T>(1)) * x * Pn - static_cast<T>(n) * Pnm1) / static_cast<T>(n + 1);
    //Coefficient1(n, x) * Pn - Coefficient2(n, x) * Pnm1;
}

template <typename T>
T LegendrePolynomial<T>::legendre_p(int const n, T const x, bool const second)
{
    // Error handling:
    if ((x < -1) || (x > 1))
    {
        UMUQFAIL("The Legendre Polynomial is defined for  -1 <= x <= 1, but got x =", x);
    }

    T P0;
    T P1;
    if (second)
    {
        // A solution of the second kind (Q):
        P0 = (std::log1p(x) - std::log1p(-x)) / static_cast<T>(2);
        P1 = x * P0 - static_cast<T>(1);
    }
    else
    {
        // A solution of the first kind (P):
        P0 = static_cast<T>(1);
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

template <typename T>
T *LegendrePolynomial<T>::legendre_p_array(int const n, T const x, bool const second)
{
    // Error handling:
    if ((x < -1) || (x > 1))
    {
        UMUQFAIL("The Legendre Polynomial is defined for  -1 <= x <= 1, but got x =", x);
    }

    T *results = nullptr;

    try
    {
        results = new T[n + 1];
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }

    T P0;
    T P1;
    if (second)
    {
        // A solution of the second kind (Q):
        P0 = (std::log1p(x) - std::log1p(-x)) / static_cast<T>(2);
        P1 = x * P0 - static_cast<T>(1);
    }
    else
    {
        // A solution of the first kind (P):
        P0 = static_cast<T>(1);
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

template <typename T>
int *LegendrePolynomial<T>::monomialBasis()
{
    if (alpha)
    {
        return alpha.get();
    }
    else
    {
        int const N = nDim * monomialSize;

        std::vector<int> x(nDim, 0);

        try
        {
            alpha.reset(new int[N]);
        }
        catch (std::bad_alloc &e)
        {
            UMUQFAILRETURNNULL("Failed to allocate memory!");
        }

        int n(0);

        for (;;)
        {
            for (int j = nDim - 1; j >= 0; j--, n++)
            {
                alpha[n] = x[j];
            }

            if (x[0] == Order)
            {
                return alpha.get();
            }

            if (!graded_reverse_lexicographic_order(x.data()))
            {
                return nullptr;
            }
        }

        return alpha.get();
    }
}

template <typename T>
int LegendrePolynomial<T>::monomialValue(T const *x, T *&value)
{
    if (!alpha)
    {
        //Have to create monomial sequence
        int *tmp = monomialBasis();

        if (tmp == nullptr)
        {
            UMUQWARNING("Something went wrong in creating monomial sequence!");
            return 0;
        }
    }

    if (value == nullptr)
    {
        try
        {
            value = new T[monomialSize];
        }
        catch (std::bad_alloc &Exponents)
        {
            UMUQWARNING("Failed to allocate memory!");
            return 0;
        }
    }

    std::vector<T *> legendrePolynomialsValues(nDim);

    for (int j = 0; j < nDim; j++)
    {
        legendrePolynomialsValues[j] = this->legendre_p_array(Order, x[j]);
    }

    for (int i = 0, k = 0; i < monomialSize; i++)
    {
        T v = static_cast<T>(1);
        for (int j = 0; j < nDim; j++, k++)
        {
            int const l = alpha[k];
            v *= legendrePolynomialsValues[j][l];
        }
        value[i] = v;
    }

    for (int j = 0; j < nDim; j++)
    {
        delete[] legendrePolynomialsValues[j];
    }

    return monomialSize;
}

template <typename T>
inline int LegendrePolynomial<T>::monomialsize() const
{
    return monomialSize;
}

template <typename T>
inline int LegendrePolynomial<T>::dim() const
{
    return nDim;
}

template <typename T>
inline int LegendrePolynomial<T>::order() const
{
    return Order;
}

template <typename T>
bool LegendrePolynomial<T>::graded_reverse_lexicographic_order(int *x)
{
    if (Order == 0)
    {
        return true;
    }

    int asum = std::accumulate(x, x + nDim, 0);

    if (asum < 0)
    {
        UMUQFAILRETURN("Input sums < 0!");
    }

    if (Order < asum)
    {
        UMUQFAILRETURN("Input sums > maximum degree r!");
    }

    if (x[0] == Order)
    {
        x[0] = 0;
        x[nDim - 1] = 0;
    }
    else
    {
        int i;
        int tmp;

        // Seeking the first index in which x > 0.
        int j = 0;
        for (i = 1; i < nDim; i++)
        {
            if (x[i] > 0)
            {
                j = i;
                break;
            }
        }

        if (j == 0)
        {
            tmp = x[0];
            x[0] = 0;
            x[nDim - 1] = tmp + 1;
        }
        else if (j < nDim - 1)
        {
            x[j] = x[j] - 1;
            tmp = x[0] + 1;
            x[0] = 0;
            x[j - 1] = x[j - 1] + tmp;
        }
        else
        {
            tmp = x[0];
            x[0] = 0;
            x[j - 1] = tmp + 1;
            x[j] = x[j] - 1;
        }
    }
    return true;
}

} // namespace umuq

#endif // UMUQ_POLYNOMIAL

// template <class T>
// void uncheckedNonZeroLegendrePolynomialsCoefficientsExponents(unsigned int const n, T *Coefficients, int *Exponents)
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