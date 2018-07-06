#ifndef UMUQ_EXPONENTIALDISTRIBUTION_H
#define UMUQ_EXPONENTIALDISTRIBUTION_H

#include "densityfunction.hpp"

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
    explicit exponentialDistribution(T const mu);

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
    inline T f(T const x);

    /*!
     * \brief Log of exponential distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T lf(T const x);
};

/*!
 * \brief Construct a new exponential distribution object
 * 
 * \param mu Mean, \f$ \mu \f$
 */
template <typename T>
explicit exponentialDistribution<T>::exponentialDistribution(T const mu) : densityFunction<T, exponentialDistribution<T>>(&mu, 1, "exponential") {}

/*!
 * \brief Exponential distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T>
inline T exponentialDistribution<T>::f(T const x)
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
template <typename T>
inline T exponentialDistribution<T>::lf(T const x)
{
    return x < this->params[0] ? std::numeric_limits<T>::infinity() : -std::log(this->params[0] - x / this->params[0]);
}

#endif //UMUQ_EXPONENTIALDISTRIBUTION_H
