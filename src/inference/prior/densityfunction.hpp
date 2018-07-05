#ifndef UMUQ_DENSITYFUNCTION_H
#define UMUQ_DENSITYFUNCTION_H

#include "../../core/core.hpp"

/*! \class densityFunction
 * \brief Density function class 
 * CRTP pattern
 * 
 * \tparam T   Data type
 * \tparam TD  Inheriting from a template class, where the derived class itself as a template parameter of the base class
 */
template <typename T, class TD>
class densityFunction
{
  public:
	/*!
     * \brief Construct a new density Function object
     * 
     */
	densityFunction() : name(""), numParams(0) {}

	/*!
     * \brief Construct a new density Function object
     * 
     * \param NumParams  Number of parameters
     */
	densityFunction(T *Params, int const NumParams, const char *Name = "") : name(Name),
																			 numParams(NumParams),
																			 params(Params, Params + NumParams)
	{
	}

	/*!
     * \brief reset the number of parameters and its argument
     * 
     * \param NumParams Number of parameters 
     * 
     * \return true 
     * \return false 
     */
	bool reset(T *Params, int const NumParams, const char *Name = "")
	{
		name = std::string(Name);
		numParams = NumParams;
		try
		{
			params.resize(numParams);
		}
		catch (...)
		{
			UMUQFAILRETURN("Failed to resize memory!")
		}
		std::copy(Params, Params + NumParams, params.data());
	}

	/*!
     * \brief Density function
     * 
     * \param x Input value
     * 
     * \returns Density function value 
     */
	inline T f(T const x)
	{
		return static_cast<TD *>(this)->f(x);
	}

	inline T f(T const *x)
	{
		return static_cast<TD *>(this)->f(x);
	}

	/*!
     * \brief Log of density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
	inline T lf(T const x)
	{
		return static_cast<TD *>(this)->lf(x);
	}

	inline T lf(T const *x)
	{
		return static_cast<TD *>(this)->lf(x);
	}

  public:
	// Name of the function
	std::string name;

	// Number of parameters
	int numParams;

	// Denisty function parameters
	std::vector<T> params;

  private:
	friend TD;
};

/*! \class uniformDistribution
 * \brief Flat (Uniform) distribution function
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a uniform distribution 
 * from \f$ \[a \cdots b\] \f$, 
 * using: 
 * \f[
 * p(x)= \left\{\begin{matrix}
 * 1/(b-a)  &a \leqslant  x < b \\ 
 *  0       &otherwise
 * \end{matrix}\right.
 * \f]
 * 
 * \tparam T Data type
 */
template <typename T>
class uniformDistribution : public densityFunction<T, uniformDistribution<T>>
{
  public:
	/*!
     * \brief Construct a new uniform Distribution object
     * 
     * \param a Lower bound
     * \param b Upper bound
     */
	uniformDistribution(T const a, T const b) : densityFunction<T, uniformDistribution<T>>(std::vector<T>{a, b}.data(), 2, "uniform") {}

	/*!
	 * \brief Destroy the uniform Distribution object
	 * 
	 */
	~uniformDistribution() {}

	/*!
     * \brief Uniform distribution density function
     * 
     * \param x Input value
     * 
     * \returns Density function value 
     */
	inline T f(T const x)
	{
		return (x < this->params[1] && x >= this->params[0]) ? static_cast<T>(1) / (this->params[1] - this->params[0]) : T{};
	}

	/*!
     * \brief Log of Uniform distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
	inline T lf(T const x)
	{
		return (x < this->params[1] && x >= this->params[0]) ? -std::log(this->params[1] - this->params[0]) : std::numeric_limits<T>::infinity();
	}
};

/*! \class gaussianDistribution
 * \brief The Gaussian distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a Gaussian 
 * distribution with standard deviation \f$ \sigma \f$
 * using: 
 * \f[
 * p(x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(-\frac{\left(x - \mu \right)^2}{2\sigma^2}\right)}
 * \f]
 * 
 * \tparam T Data type
 */
template <typename T>
class gaussianDistribution : public densityFunction<T, gaussianDistribution<T>>
{
  public:
	/*!
     * \brief Construct a new gaussian Distribution object
     * 
     * \param mu     Mean, \f$ \mu \f$
     * \param sigma  Standard deviation \f$ \sigma \f$
     */
	gaussianDistribution(T const mu, T const sigma) : densityFunction<T, gaussianDistribution<T>>(std::vector<T>{mu, sigma}.data(), 2, "gaussian") {}

	/*!
	 * \brief Destroy the gaussian Distribution object
	 * 
	 */
	~gaussianDistribution() {}

	/*!
     * \brief Gaussian Distribution density function
     * 
     * \param x Input value
     * 
     * \returns Density function value 
     */
	inline T f(T const x)
	{
		T const xSigma = x - this->params[0] / this->params[1];
		return static_cast<T>(1) / (std::sqrt(M_2PI) * this->params[1]) * std::exp(-xSigma * xSigma / static_cast<T>(2));
	}

	/*!
     * \brief Log of Gaussian Distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
	inline T lf(T const x)
	{
		T const xSigma = x - this->params[0] / this->params[1];
		return -0.5 * M_L2PI - std::log(this->params[1]) - 0.5 * xSigma * xSigma;
	}
};

/*! \class exponentialDistribution
 * \brief The exponential distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for an 
 * exponential distribution with mean \f$ \mu \f$
 * using: 
 * \f[
 * p(x)=\frac{1}{\mu}e^{\left(-\frac{x}{\mu}\right)}
 * \f]
 * 
 * \tparam T Data type
 */
template <typename T>
class exponentialDistribution : public densityFunction<T, exponentialDistribution<T>>
{
  public:
	/*!
     * \brief Construct a new exponential distribution object
     * 
     * \param mu Mean, \f$ \mu \f$
     */
	exponentialDistribution(T const mu) : densityFunction<T, exponentialDistribution<T>>(&mu, 1, "exponential") {}

	/*!
	 * \brief Destroy the exponential distribution object
	 * 
	 */
	~exponentialDistribution() {}

	/*!
     * \brief Exponential distribution density function
     * 
     * \param x Input value
     * 
     * \returns Density function value 
     */
	inline T f(T const x)
	{
		return x < T{} ? T{} : std::exp(-x / this->params[0]) / this->params[0];
	}

	/*!
     * \brief Log of exponential distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
	inline T lf(T const x)
	{
		return x < this->params[0] ? std::numeric_limits<T>::infinity() : -std::log(this->params[0] - x / this->params[0]);
	}
};

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
class gammaDistribution : public densityFunction<T, gammaDistribution<T>>
{
  public:
	/*!
     * \brief Construct a new Gamma distribution object
     * 
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     */
	gammaDistribution(T const alpha, T const beta = T{1}) : densityFunction<T, gammaDistribution<T>>(std::vector<T>{alpha, beta}.data(), 2, "gamma") {}

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
	inline T f(T const x)
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
	inline T lf(T const x)
	{
		return x < T{} ? -std::numeric_limits<T>::infinity() : -std::lgamma(this->params[0]) - this->params[0] * std::log(this->params[1]) + (this->params[0] - static_cast<T>(1)) * std::log(x) - x / this->params[1];
	}
};

#endif // UMUQ_DENSITYFUNCTION_H