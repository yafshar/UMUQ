#ifndef UMUQ_GAUSSIANDISTRIBUTION_H
#define UMUQ_GAUSSIANDISTRIBUTION_H

#include "../function/densityfunction.hpp"

/*! \class gaussianDistribution
 * \brief The Gaussian distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a Gaussian 
 * distribution wiuh standard deviation \f$ \sigma \f$
 * using: 
 * \f[
 * p(x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(-\frac{\left(x - \mu \right)^2}{2\sigma^2}\right)}
 * \f]
 * 
 * \tparam T Data type
 */
template <typename T, class V = T const *>
class gaussianDistribution : public densityFunction<T, std::function<T(V)>>
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
     * \brief Construct a new gaussian Distribution object
     * 
     * \param mu     Mean, \f$ \mu \f$
     * \param sigma  Standard deviation \f$ \sigma \f$
     * \param n      Total number of Mean + Standard deviation inputs
     */
    gaussianDistribution(T const *mu, T const *sigma, int const n);

    /*!
     * \brief Destroy the gaussian Distribution object
     * 
     */
    ~gaussianDistribution() {}

    /*!
     * \brief Gaussian Distribution density function
     * 
     * \param x  Input value
     * 
     * \returns Density function value 
     */
    inline T gaussianDistribution_f(T const *x);

    /*!
     * \brief Log of Gaussian Distribution density function
     * 
     * \param x  Input value
     * 
     * \returns  Log of density function value 
     */
    inline T gaussianDistribution_lf(T const *x);
};

/*!
 * \brief Construct a new gaussian Distribution object
 * 
 * \param mu     Mean, \f$ \mu \f$
 * \param sigma  Standard deviation \f$ \sigma \f$
 */
template <typename T, class V>
gaussianDistribution<T, V>::gaussianDistribution(T const mu, T const sigma) : densityFunction<T, std::function<T(V)>>(&mu, &sigma, 2, "gaussian")
{
    this->f = std::bind(&gaussianDistribution<T, V>::gaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gaussianDistribution<T, V>::gaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
gaussianDistribution<T, V>::gaussianDistribution(T const *mu, T const *sigma, int const n) : densityFunction<T, std::function<T(V)>>(mu, sigma, n, "gaussian")
{
    if (n % 2 != 0)
    {
        UMUQFAIL("Wrong number of inputs!");
    }
    this->f = std::bind(&gaussianDistribution<T, V>::gaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gaussianDistribution<T, V>::gaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Gaussian Distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T, class V>
inline T gaussianDistribution<T, V>::gaussianDistribution_f(T const *x)
{
    T sum(1);
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        T const xSigma = (x[i] - this->params[k]) / this->params[k + 1];
        sum *= static_cast<T>(1) / (M_S2PI * this->params[k + 1]) * std::exp(-0.5 * xSigma * xSigma);
    }
    return sum;
}

/*!
 * \brief Log of Gaussian Distribution density function
 * 
 * \param x Input value
 * 
 * \returns  Log of density function value 
 */
template <typename T, class V>
inline T gaussianDistribution<T, V>::gaussianDistribution_lf(T const *x)
{
    T sum(0);
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        T const xSigma = (x[i] - this->params[k]) / this->params[k + 1];
        sum += -0.5 * M_L2PI - std::log(this->params[k + 1]) - 0.5 * xSigma * xSigma;
    }
    return sum;
}

#endif //UMUQ_GAUSSIANDISTRIBUTION_H