#ifndef UMUQ_GAUSSIANDISTRIBUTION_H
#define UMUQ_GAUSSIANDISTRIBUTION_H

#include "densityfunction.hpp"

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
    gaussianDistribution(T const mu, T const sigma);

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
    inline T f(T const x);

    /*!
     * \brief Log of Gaussian Distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T lf(T const x);
};

/*!
 * \brief Construct a new gaussian Distribution object
 * 
 * \param mu     Mean, \f$ \mu \f$
 * \param sigma  Standard deviation \f$ \sigma \f$
 */
template <typename T>
gaussianDistribution<T>::gaussianDistribution(T const mu, T const sigma) : densityFunction<T, gaussianDistribution<T>>(std::vector<T>{mu, sigma}.data(), 2, "gaussian") {}
/*!
 * \brief Gaussian Distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T>
inline T gaussianDistribution<T>::f(T const x)
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
template <typename T>
inline T gaussianDistribution<T>::lf(T const x)
{
    T const xSigma = x - this->params[0] / this->params[1];
    return -0.5 * M_L2PI - std::log(this->params[1]) - 0.5 * xSigma * xSigma;
}

#endif // UMUQ_GAUSSIANDISTRIBUTION_H
