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

template <typename T, class V = T const *>
class gammaDistribution : public densityFunction<T, std::function<T(V)>>
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
     * \brief Construct a new gamma Distribution object
     * 
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     * \param n      Total number of alpha + beta inputs
     */
    gammaDistribution(T const *alpha, T const *beta, int const n);

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
    inline T gammaDistribution_f(T const *x);

    /*!
     * \brief Log of Gamma distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T gammaDistribution_lf(T const *x);
};

/*!
 * \brief Construct a new Gamma distribution object
 * 
 * \param alpha  Shape parameter \f$\alpha\f$
 * \param beta   Scale parameter \f$ beta\f$
 */
template <typename T, class V>
gammaDistribution<T, V>::gammaDistribution(T const alpha, T const beta) : densityFunction<T, std::function<T(V)>>(&alpha, &beta, 2, "gamma")
{
    this->f = std::bind(&gammaDistribution<T, V>::gammaDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gammaDistribution<T, V>::gammaDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
gammaDistribution<T, V>::gammaDistribution(T const *alpha, T const *beta, int const n) : densityFunction<T, std::function<T(V)>>(alpha, beta, n, "gamma")
{
    if (n % 2 != 0)
    {
        UMUQFAIL("Wrong number of inputs!")
    }
    this->f = std::bind(&gammaDistribution<T, V>::gammaDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gammaDistribution<T, V>::gammaDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Gamma distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T, class V>
inline T gammaDistribution<T, V>::gammaDistribution_f(T const *x)
{
    T sum(1);
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        if (x[i] < T{})
        {
            return T{};
        }
        else if (x[i] == T{})
        {
            if (this->params[k] == static_cast<T>(1))
            {
                sum *= static_cast<T>(1) / this->params[k + 1];
                continue;
            }
            else
            {
                return T{};
            }
        }
        else if (this->params[k] == static_cast<T>(1))
        {
            sum *= std::exp(-x[i] / this->params[k + 1]) / this->params[k + 1];
        }
        else
        {
            sum *= std::exp((this->params[k] - static_cast<T>(1)) * std::log(x[i] / this->params[k + 1]) - x[i] / this->params[k + 1] - std::lgamma(this->params[k])) / this->params[k + 1];
        }
    }
    return sum;
}

/*!
 * \brief Log of Gamma distribution density function
 * 
 * \param x Input value
 * 
 * \returns  Log of density function value 
 */
template <typename T, class V>
inline T gammaDistribution<T, V>::gammaDistribution_lf(T const *x)
{
    for (std::size_t i = 0; i < this->numParams / 2; i++)
    {
        if (x[i] < T{})
        {
            return -std::numeric_limits<T>::infinity();
        }
    }
    T sum(0);
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        sum += -std::lgamma(this->params[k]) - this->params[k] * std::log(this->params[k + 1]) + (this->params[k] - static_cast<T>(1)) * std::log(x[i]) - x[i] / this->params[k + 1];
    }
    return sum;
}

#endif // UMUQ_GAMMADISTRIBUTION
