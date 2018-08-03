#ifndef UMUQ_GAMMADISTRIBUTION_H
#define UMUQ_GAMMADISTRIBUTION_H

#include "../function/densityfunction.hpp"

/*! \class gammaDistribution
 * \brief The Gamma distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a
 * Gamma distribution with shape parameter \f$\alpha\f$ and scale parameter \f$ beta\f$.
 * The scale parameter, \f$ beta\f$, is optional and defaults to \f$ beta = 1\f$.
 * using: 
 * \f[
 * p(x)=\frac{1}{\Gamma (\alpha) \beta^\alpha}x^{\alpha-1}e^{\frac{-x}{\beta}}
 * \f]
 * 
 * Use the Gamma distribution with \f$\alpha > 1\f$ if you have a sharp lower bound of zero but no 
 * sharp upper bound, a single mode, and a positive skew. The Gamma distribution is especially appropriate 
 * when encoding arrival times for sets of events. A Gamma distribution with a large value for \f$\alpha\f$ 
 * is also useful when you wish to use a bell-shaped curve for a positive-only quantity.
 * 
 * 
 * \tparam T Data type
 */
template <typename T>
class gammaDistribution : public densityFunction<T, FUN_x<T>>
{
  public:
	/*!
     * \brief Construct a new Gamma distribution object
     * 
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     */
	gammaDistribution(T const alpha, T const beta = T{1});

	/*!
	 * \brief Destroy the Gamma distribution object
	 * 
	 */
	~gammaDistribution() {}

	/*!
     * \brief Gamma distribution density function
     * 
     * \param x Input value
     * 
     * \returns Density function value 
     */
	inline T gammaDistribution_f(T const x);

	/*!
     * \brief Log of Gamma distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
	inline T gammaDistribution_lf(T const x);
};

/*!
 * \brief Construct a new Gamma distribution object
 * 
 * \param alpha  Shape parameter \f$\alpha\f$
 * \param beta   Scale parameter \f$ beta\f$
 */
template <typename T>
gammaDistribution<T>::gammaDistribution(T const alpha, T const beta) : densityFunction<T, FUN_x<T>>(std::vector<T>{alpha, beta}.data(), 2, "gamma")
{
	this->f = std::bind(&gammaDistribution<T>::gammaDistribution_f, this, std::placeholders::_1);
	this->lf = std::bind(&gammaDistribution<T>::gammaDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Gamma distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T>
inline T gammaDistribution<T>::gammaDistribution_f(T const x)
{
	if (x < T{})
	{
		return T{};
	}
	else if (x == T{})
	{
		if (this->params[0] == static_cast<T>(1))
		{
			return static_cast<T>(1) / this->params[1];
		}
		else
		{
			return T{};
		}
	}
	else if (this->params[0] == static_cast<T>(1))
	{
		return std::exp(-x / this->params[1]) / this->params[1];
	}
	else
	{
		return std::exp((this->params[0] - static_cast<T>(1)) *
							std::log(x / this->params[1]) -
						x / this->params[1] - std::lgamma(this->params[0])) /
			   this->params[1];
	}
}

/*!
 * \brief Log of Gamma distribution density function
 * 
 * \param x Input value
 * 
 * \returns  Log of density function value 
 */
template <typename T>
inline T gammaDistribution<T>::gammaDistribution_lf(T const x)
{
	return x < T{} ? -std::numeric_limits<T>::infinity() : -std::lgamma(this->params[0]) - this->params[0] * std::log(this->params[1]) + (this->params[0] - static_cast<T>(1)) * std::log(x) - x / this->params[1];
}

#endif // UMUQ_GAMMADISTRIBUTION_H
