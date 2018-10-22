#ifndef UMUQ_MONOMIAL_H
#define UMUQ_MONOMIAL_H

#include "./polynomialbase.hpp"

namespace umuq
{

inline namespace polynomials
{

/*! \class polynomial
 * \ingroup Polynomials_Module
 *
 * \brief Multivariate monomials with the degree of \b r in a space of \b d dimensions.
 *
 * A (univariate) monomial in \f$ 1 \f$ variable \f$ x \f$ is simply any (non-negative integer) 
 * power of \f$ x \f$:<br>
 * \f$  1, x, x^2, x^3, \cdots, x^r \f$<br>
 * The highest exponent of \f$ x \f$ is termed the \b degree of the monomial.
 * If several variables are considered, say, \f$ x,~y,~\text{and}~z \f$ then each can be given an exponent, 
 * so that any monomial is of the form \f$ x^ay^bz^c\f$ with \f$ a,~b,~\text{and}~c \f$ non-negative integers 
 * (taking note that any exponent 0 makes the corresponding factor equal to 1).
 */
template <typename T>
class polynomial : public polynomialBase<T>
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
     * \param  x      The coordinates of the evaluation points
     * \param  value  The (monomial value) array value of the monomial at point x
     * 
     * \returns The size of the monomial array
     */
	int monomialValue(T const *x, T *&value);

  private:
	/*!
     * \brief Delete a polynomial object copy construction
     * 
     * Make it noncopyable.
     */
	polynomial(polynomial<T> const &) = delete;

	/*!
     * \brief Delete a polynomial object assignment
     * 
     * Make it nonassignable
     * 
     * \returns polynomial<T>& 
     */
	polynomial<T> &operator=(polynomial<T> const &) = delete;
};

template <typename T>
polynomial<T>::polynomial(int const dim, int const PolynomialOrder) : polynomialBase<T>(dim, PolynomialOrder) {}

template <typename T>
int *polynomial<T>::monomialBasis()
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

template <typename T>
int polynomial<T>::monomialValue(T const *x, T *&value)
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
			value = new T[this->monomialSize];
		}
		catch (...)
		{
			UMUQFAIL("Failed to allocate memory!");
		}
	}

	for (int i = 0, k = 0; i < this->monomialSize; i++)
	{
		T v = static_cast<T>(1);
		for (int j = 0; j < this->nDim; j++, k++)
		{
			v *= std::pow(x[j], this->alpha[k]);
		}
		value[i] = v;
	}

	return this->monomialSize;
}

} // namespace polynomials
} // namespace umuq

#endif // UMUQ_POLYNOMIAL
