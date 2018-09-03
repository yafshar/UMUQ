#ifndef UMUQ_EXPONENTIALDISTRIBUTION_H
#define UMUQ_EXPONENTIALDISTRIBUTION_H

#include "../function/densityfunction.hpp"

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
template <typename T, class V = T const *>
class exponentialDistribution : public densityFunction<T, std::function<T(V)>>
{
  public:
    /*!
     * \brief Construct a new exponential distribution object
     * 
     * \param mu Mean, \f$ \mu \f$
     */
    explicit exponentialDistribution(T const mu);

    /*!
     * \brief Construct a new exponential Distribution object
     * 
     * \param mu Mean, \f$ \mu \f$
     * \param n  Number of input
     */
    explicit exponentialDistribution(T const *mu, int const n);

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
    inline T exponentialDistribution_f(T const *x);

    /*!
     * \brief Log of exponential distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T exponentialDistribution_lf(T const *x);

  private:
    /*!
     * \brief Construct a new exponential Distribution object
     * 
     */
    exponentialDistribution() = delete;
};

/*!
 * \brief Construct a new exponential distribution object
 * 
 * \param mu Mean, \f$ \mu \f$
 */
template <typename T, class V>
exponentialDistribution<T, V>::exponentialDistribution(T const mu) : densityFunction<T, std::function<T(V)>>(&mu, 1, "exponential")
{
    this->f = std::bind(&exponentialDistribution<T, V>::exponentialDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&exponentialDistribution<T, V>::exponentialDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
exponentialDistribution<T, V>::exponentialDistribution(T const *mu, int const n) : densityFunction<T, std::function<T(V)>>(mu, n, "exponential")
{
    this->f = std::bind(&exponentialDistribution<T, V>::exponentialDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&exponentialDistribution<T, V>::exponentialDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Exponential distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T, class V>
inline T exponentialDistribution<T, V>::exponentialDistribution_f(T const *x)
{
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        if (x[i] < T{})
        {
            return T{};
        }
    }
    T sum(1);
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        sum *= std::exp(-x[i] / this->params[i]) / this->params[i];
    }
    return sum;
}

/*!
 * \brief Log of exponential distribution density function
 * 
 * \param x Input value
 * 
 * \returns  Log of density function value 
 */
template <typename T, class V>
inline T exponentialDistribution<T, V>::exponentialDistribution_lf(T const *x)
{
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        if (x[i] < T{})
        {
            return std::numeric_limits<T>::infinity();
        }
    }
    T sum(0);
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        sum -= std::log(this->params[i] - x[i] / this->params[i]);
    }
    return sum;
}

#endif //UMUQ_EXPONENTIALDISTRIBUTION
