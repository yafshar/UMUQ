#ifndef UMUQ_POLYNOMIALBASE_H
#define UMUQ_POLYNOMIALBASE_H

#include "core/core.hpp"
#include "numerics/factorial.hpp"

#include <cmath>

#include <vector>
#include <type_traits>
#include <utility>
#include <numeric>
#include <memory>

namespace umuq
{

inline namespace polynomials
{

/*! \class polynomialBase
 * \ingroup Polynomials_Module
 *
 * \brief This is the base class for different multivariate monomials with the degree of \b r in a space of \b d dimensions.
 * \sa umuq::polynomials::polynomial
 *
 * \tparam DataType Data type
 *
 * In %UMUQ we consider a monomial, as a power product, which is a product of powers of variables
 * with nonnegative integer exponents.<br>
 *
 * A (univariate) monomial in \f$ 1 \f$ variable \f$ x \f$ is simply any (non-negative integer)
 * power of \f$ x \f$:<br>
 *
 * \f$  1, x, x^2, x^3, \cdots, x^r \f$<br>
 *
 * The highest exponent of \f$ x \f$ is termed the \b degree of the monomial.<br>
 * If several variables are considered, say, \f$ x,~y,~\text{and}~z \f$ then each can be given an exponent,
 * so that any monomial is of the form \f$ x^ay^bz^c\f$ with \f$ a,~b,~\text{and}~c \f$ non-negative integers
 * (taking note that any exponent 0 makes the corresponding factor equal to 1).
 * \sa umuq::polynomials::polynomial
 *
 * In %UMUQ, we replace the monomials by polynomials of different types, especially when polynomial \f$ P_n(x) \f$
 * satisfies a recurrence relation. \sa umuq::polynomials::PolynomialTypes.
 *
 * This way, a (univariate) monomial in \f$ 1 \f$ variable \f$ x \f$ can simply be replacd by a
 * (univariate) polynomial of \f$ P_n(x) \f$:
 *
 * \f$ P_0(x), P_1(x), P_2(x), P_3(x), \cdots, P_r(x) \f$
 *
 * If we replace the monomials in the Vandermonde matrix by different polynomials, the resulting matrix is a
 * Vandermonde-like matrix.
 */

template <typename DataType>
class polynomialBase
{
  public:
    /*!
     * \brief Construct a new polynomialBase object
     *
     * \param dim              Dimension
     * \param PolynomialOrder  Polynomial order (the default order or degree of r in a space of dim dimensions is 2)
     */
    polynomialBase(int const dim, int const PolynomialOrder = 2);

    /*!
     * \brief Reset the dimension of the problem and rhe desired order
     *
     * Reset the dimension and order values to the new ones
     *
     * \param dim              New Dimension
     * \param PolynomialOrder  New Order (the default order or degree of \b r in a space of \b dim dimensions is 2)
     */
    void reset(int const dim, int const PolynomialOrder = 2);

    /*!
     * \brief Computes the [binomial coefficient](https://en.wikipedia.org/wiki/Binomial_coefficient)
     * \f$ C(n, k) \f$.
     *
     * -# A binomial coefficient \f$ C(n, k) \f$ can be defined as the coefficient of \f$ x ^ k \f$ in
     * the expansion of \f$ (1 + x) ^ n. \f$ <br>
     * -# A binomial coefficient \f$ C(n, k) \f$ also gives the number of ways, disregarding order,
     * that k objects can be chosen from among n objects; <br>
     * More formally, the number of k-element subsets (or k-combinations) of an n-element set.
     *
     * The formula used is:
     *
     * \f$ C(n, k) = \frac{n!}{ n! (n-k)! } \f$
     *
     * \param n  Input parameter
     * \param k  Input parameter
     *
     * \returns The binomial coefficient \f$ C(n, k) \f$
     */
    int binomialCoefficient(int const n, int const k);

    /*!
     * \brief Here, \f$\alpha=\f$ all the monomials in a \b d dimensional space, with total degree \b r.
     *
     * \returns A pointer to monomial sequence
     */
    virtual int *monomialBasis();

    /*!
     * \brief Evaluates a monomial at a point x.
     *
     * \param  x      The abscissa values. (The coordinates of the evaluation points)
     * \param  value  The (monomial value) array value of the monomial at point x
     *
     * \returns The size of the monomial array
     */
    virtual int monomialValue(DataType const *x, DataType *&value);

    /*!
     * \brief Evaluates a monomial at a point x.
     *
     * \param  x      The abscissa values. (The coordinates of the evaluation points)
     * \param  value  The (monomial value) array value of the monomial at point x
     *
     * \returns The size of the monomial array
     */
    virtual int monomialValue(DataType const *x, std::vector<DataType> &value);

    /*!
     * \brief Evaluates monomial derivatives at origin.
     *
     * \param beta   In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right). \f$<br>
     *               Notation for partial derivatives:<br>
     *               \f$  D^\beta = \frac{\partial^{|\beta|}}{\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \f$
     * \param value  The (monomial derivative value) array value of the monomial derivatives at zero point
     *
     * \returns int The size of the monomial array
     */
    virtual int monomialDerivative(int const *beta, DataType *&value);

    /*!
     * \brief Evaluates monomial derivatives at origin.
     *
     * \param beta   In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right). \f$<br>
     *               Notation for partial derivatives:<br>
     *               \f$  D^\beta = \frac{\partial^{|\beta|}}{\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \f$
     * \param value  The (monomial derivative value) array value of the monomial derivatives at zero point
     *
     * \returns int The size of the monomial array
     */
    virtual int monomialDerivative(int const *beta, std::vector<DataType> &value);

    /*!
     * \brief Get the monomial size
     *
     * \return Monomial size
     */
    inline int monomialsize() const;

    /*!
     * \brief Get the dimension
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

  protected:
    /*!
     * \brief Use a reverse lexicographic order for next monomial, degrees between 0 and \b r
     *  all monomials in a \b d dimensional space, with order of accuracy \b r.
     *
     * \param x  Current monomial on input and next monomial on the output (last value in the sequence is r).
     */
    bool graded_reverse_lexicographic_order(int *x);

  protected:
    /*!
     * \brief Delete a polynomialBase object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    polynomialBase(polynomialBase<DataType> const &) = delete;

    /*!
     * \brief Delete a polynomialBase object assignment
     *
     * Avoiding implicit copy assignment.
     */
    polynomialBase<DataType> &operator=(polynomialBase<DataType> const &) = delete;

  protected:
    //! Dimension
    int nDim;

    //! Order of accuracy
    int Order;

    //! The size of the monomial array
    int monomialSize;

    //! Array of monomial sequence
    std::unique_ptr<int[]> alpha;
};

template <typename DataType>
polynomialBase<DataType>::polynomialBase(int const dim, int const PolynomialOrder) : nDim(dim), Order(PolynomialOrder)
{
    if (nDim <= 0)
    {
        UMUQFAIL("Can not have dimension ", nDim, " <= 0!");
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

template <typename DataType>
void polynomialBase<DataType>::reset(int const dim, int const PolynomialOrder)
{
    nDim = dim;
    if (nDim <= 0)
    {
        UMUQFAIL("Can not have dimension ", nDim, " <= 0!");
    }

    Order = PolynomialOrder;
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

template <typename DataType>
int polynomialBase<DataType>::binomialCoefficient(int const n, int const k)
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

template <typename DataType>
int *polynomialBase<DataType>::monomialBasis()
{
    UMUQFAIL("This is a virtual method in the base class and not implemented on purpose!");
    return nullptr;
}

template <typename DataType>
int polynomialBase<DataType>::monomialValue(DataType const *x, DataType *&value)
{
    UMUQFAIL("This is a virtual method in the base class and not implemented on purpose!");
    return -1;
}

template <typename DataType>
int polynomialBase<DataType>::monomialValue(DataType const *x, std::vector<DataType> &value)
{
    UMUQFAIL("This is a virtual method in the base class and not implemented on purpose!");
    return -1;
}

template <typename DataType>
int polynomialBase<DataType>::monomialDerivative(int const *beta, DataType *&value)
{
    UMUQFAIL("This is a virtual method in the base class and not implemented on purpose!");
    return -1;
}

template <typename DataType>
int polynomialBase<DataType>::monomialDerivative(int const *beta, std::vector<DataType> &value)
{
    UMUQFAIL("This is a virtual method in the base class and not implemented on purpose!");
    return -1;
}

template <typename DataType>
inline int polynomialBase<DataType>::monomialsize() const
{
    return monomialSize;
}

template <typename DataType>
inline int polynomialBase<DataType>::dim() const
{
    return nDim;
}

template <typename DataType>
inline int polynomialBase<DataType>::order() const
{
    return Order;
}

template <typename DataType>
bool polynomialBase<DataType>::graded_reverse_lexicographic_order(int *x)
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

} // namespace polynomials
} // namespace umuq

#endif // UMUQ_POLYNOMIALBASE
