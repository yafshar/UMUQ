#ifndef UMUQ_MONOMIAL_H
#define UMUQ_MONOMIAL_H

#include "core/core.hpp"
#include "polynomialbase.hpp"

#include <cmath>
#include <cstddef>

#include <vector>
#include <algorithm>

namespace umuq
{

inline namespace polynomials
{

/*! \class polynomial
 * \ingroup Polynomials_Module
 *
 * \brief Multivariate monomials with the degree of \b r in a space of \b d dimensions.
 *
 * \tparam DataType Data type
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
 */
template <typename DataType>
class polynomial : public polynomialBase<DataType>
{
  public:
    /*!
     * \brief Construct a new polynomial object
     *
     * \param dim              Dimension
     * \param PolynomialOrder  Polynomial order (the default order or degree of r in a space of dim dimensions is 2)
     */
    polynomial(int const dim, int const PolynomialOrder = 2);

    /*!
     * \brief Move constructor, construct a new polynomial object
     *
     * \param other polynomial object
     */
    polynomial(polynomial<DataType> &&other) = default;

    /*!
     * \brief Move assignment operator
     *
     * \param other polynomial object
     *
     * \returns polynomial<DataType>& polynomial object
     */
    polynomial<DataType> &operator=(polynomial<DataType> &&other) = default;

    /*!
     * \brief Destroy the polynomial object
     *
     */
    ~polynomial() = default;

    /*!
     * \brief Here, \f$\alpha=\f$ all the monomials in a \b d dimensional space, with total degree \b r.
     *
     * For example: <br>
     * \verbatim
     *       d = 2
     *       r = 2
     *
     *       alpha[ 0],[ 1] = 0, 0 = x^0 y^0
     *       alpha[ 2],[ 3] = 1, 0 = x^1 y^0
     *       alpha[ 4],[ 5] = 0, 1 = x^0 y^1
     *       alpha[ 6],[ 7] = 2, 0 = x^2 y^0
     *       alpha[ 8],[ 9] = 1, 1 = x^1 y^1
     *       alpha[10],[11] = 0, 2 = x^0 y^2
     *
     *       monomialBasis_(d=2,r=2)   = {1,    x,   y,  x^2, xy,  y^2}
     *                           alpha = {0,0, 1,0, 0,1, 2,0, 1,1, 0,2}
     *
     *
     *       monomialBasis_(d=3,r=2)   = {1,       x,     y,     z,    x^2,  xy,    xz,   y^2,    yz,    z^2  }
     *                           alpha = {0,0,0, 1,0,0, 0,1,0, 0,0,1, 2,0,0 1,1,0, 1,0,1, 0,2,0, 0,1,1, 0,0,2 }
     *
     * \endverbatim
     *
     * \returns A pointer to monomial sequence
     */
    int *monomialBasis();

    /*!
     * \brief Evaluates a monomial at a point x.
     *
     * \param  x      The abscissa values. (The coordinates of the evaluation points)
     * \param  value  The (monomial value) array value of the monomial at point x
     *
     * \returns The size of the monomial array
     */
    int monomialValue(DataType const *x, DataType *&value);

    /*!
     * \brief Evaluates a monomial at a point x.
     *
     * \param  x      The abscissa values. (The coordinates of the evaluation points)
     * \param  value  The (monomial value) array value of the monomial at point x
     *
     * \returns The size of the monomial array
     */
    int monomialValue(DataType const *x, std::vector<DataType> &value);

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
    int monomialDerivative(int const *beta, DataType *&value);

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
    int monomialDerivative(int const *beta, std::vector<DataType> &value);

  private:
    /*!
     * \brief Delete a polynomial object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    polynomial(polynomial<DataType> const &) = delete;

    /*!
     * \brief Delete a polynomial object assignment
     *
     * Avoiding implicit copy assignment.
     */
    polynomial<DataType> &operator=(polynomial<DataType> const &) = delete;
};

template <typename DataType>
polynomial<DataType>::polynomial(int const dim, int const PolynomialOrder) : polynomialBase<DataType>(dim, PolynomialOrder) {}

template <typename DataType>
int *polynomial<DataType>::monomialBasis()
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

template <typename DataType>
int polynomial<DataType>::monomialValue(DataType const *x, DataType *&value)
{
    if (!this->alpha)
    {
        // Have to create monomial sequence
        int *tmp = monomialBasis();

        if (tmp == nullptr)
        {
            UMUQFAIL("Something went wrong in creating monomial sequence!");
        }
    }

    if (value == nullptr)
    {
        try
        {
            value = new DataType[this->monomialSize];
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }

    for (int i = 0, k = 0; i < this->monomialSize; i++)
    {
        DataType v = static_cast<DataType>(1);
        for (int j = 0; j < this->nDim; j++, k++)
        {
            v *= std::pow(x[j], this->alpha[k]);
        }
        value[i] = v;
    }

    return this->monomialSize;
}

template <typename DataType>
int polynomial<DataType>::monomialValue(DataType const *x, std::vector<DataType> &value)
{
    if (!this->alpha)
    {
        // Have to create monomial sequence
        int *tmp = monomialBasis();

        if (tmp == nullptr)
        {
            UMUQFAIL("Something went wrong in creating monomial sequence!");
        }
    }

    if (value.size() < static_cast<std::size_t>(this->monomialSize))
    {
        value.resize(this->monomialSize);
    }

    for (int i = 0, k = 0; i < this->monomialSize; i++)
    {
        DataType v = static_cast<DataType>(1);
        for (int j = 0; j < this->nDim; j++, k++)
        {
            v *= std::pow(x[j], this->alpha[k]);
        }
        value[i] = v;
    }

    return this->monomialSize;
}

template <typename DataType>
int polynomial<DataType>::monomialDerivative(int const *beta, DataType *&value)
{
    if (!this->alpha)
    {
        // Have to create monomial sequence
        int *tmp = monomialBasis();

        if (tmp == nullptr)
        {
            UMUQFAIL("Something went wrong in creating monomial sequence!");
        }
    }

    if (value == nullptr)
    {
        try
        {
            value = new DataType[this->monomialSize];
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }

    for (int i = 0; i < this->monomialSize; i++)
    {
        int maxalpha = 0;
        for (int j = 0; j < this->nDim; j++)
        {
            maxalpha = std::max(maxalpha, std::abs(this->alpha[i] - beta[j]));
        }
        if (maxalpha)
        {
            value[i] = DataType{};
        }
        else
        {
            DataType fact = static_cast<DataType>(1);
            std::for_each(beta, beta + this->nDim, [&](int const b_j) { fact *= factorial<DataType>(b_j); });
            value[i] = fact;
        }
    }

    return this->monomialSize;
}

template <typename DataType>
int polynomial<DataType>::monomialDerivative(int const *beta, std::vector<DataType> &value)
{
    if (!this->alpha)
    {
        // Have to create monomial sequence
        int *tmp = monomialBasis();

        if (tmp == nullptr)
        {
            UMUQFAIL("Something went wrong in creating monomial sequence!");
        }
    }

    if (value.size() < static_cast<std::size_t>(this->monomialSize))
    {
        value.resize(this->monomialSize);
    }

    for (int i = 0; i < this->monomialSize; i++)
    {
        int maxalpha = 0;
        for (int j = 0; j < this->nDim; j++)
        {
            maxalpha = std::max(maxalpha, std::abs(this->alpha[i] - beta[j]));
        }
        if (maxalpha)
        {
            value[i] = DataType{};
        }
        else
        {
            DataType fact = static_cast<DataType>(1);
            std::for_each(beta, beta + this->nDim, [&](int const b_j) { fact *= factorial<DataType>(b_j); });
            value[i] = fact;
        }
    }

    return this->monomialSize;
}

} // namespace polynomials
} // namespace umuq

#endif // UMUQ_POLYNOMIAL
