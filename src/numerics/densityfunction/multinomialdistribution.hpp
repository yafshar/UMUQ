#ifndef UMUQ_MULTINOMIALDISTRIBUTION_H
#define UMUQ_MULTINOMIALDISTRIBUTION_H

#include "densityfunction.hpp"

/*! \class multinomialDistribution
 * \brief The multinomial distribution
 * The multinomial distribution models the probability of counts for rolling a k-sided die n times. 
 * For n independent trials each of which leads to a success for exactly one of k categories, with 
 * each category having a given fixed success probability, the multinomial distribution gives the 
 * probability of any particular combination of numbers of successes for the various categories.
 * 
 * Reference:
 * https://en.wikipedia.org/wiki/Multinomial_distribution
 * 
 * 
 * This class provides probability density \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
 * from a multinomial distribution with probabilities \f$p[K]\f$.
 * using: 
 * \f[
 *     Pr(X_1=n_1, \cdots, X_K=n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \cdots p_K^{n_K}
 * \f] 
 *
 * where \f$ n_1, \cdots n_K \f$ are nonnegative integers satisfying \f$ sum_{i=1}^{K} {n_i} = N\f$,
 * and \f$p = \left(p_1, \cdots, p_K\right)\f$ is a probability distribution. 
 * 
 * \tparam T Data type
 */
template <typename T>
class multinomialDistribution : public densityFunction<T, multinomialDistribution<T>>
{
  public:
    /*!
     * \brief Construct a new multinomial distribution object
     * 
     * \param  k  Vector size
     */
    multinomialDistribution(int const k);

    /*!
     * \brief Destroy the multinomial distribution object
     *
     */
    ~multinomialDistribution() {}

    /*!
     * \brief multinomial distribution density function
     * Computes the probability from the multinomial distribution
     * 
     * This function computes the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
     * from a multinomial distribution with probabilities \f$p[K]\f$.
     * 
     * \param  p       Vector of probabilities \f$ p_1, \cdots, p_k \f$ (with size of K)
     * \param  mndist  A random sample (with size of K) from the multinomial distribution 
     * 
     * \returns Density function value (the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$)
     */
    template <unsigned int>
    inline T f(T const *p, unsigned int const *mndist);

    /*!
     * \brief Log of multinomial distribution density function
     * Computes the logarithm of the probability from the multinomial distribution
     * 
     * This function computes the logarithm of the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
     * from a multinomial distribution with probabilities \f$p[K]\f$.
     * 
     * \param p       Vector of probabilities \f$ p_1, \cdots, p_k \f$ (with size of K)
     * \param mndist  A random sample (with size of K) from the multinomial distribution
     * 
     * \returns  Log of density function value (the logarithm of the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ )
     */
    template <unsigned int>
    inline T lf(T const *p, unsigned int const *mndist);
};

/*!
 * \brief Construct a new multinomial distribution object
 * 
 * \param  k  Vector size
 */
template <typename T>
multinomialDistribution<T>::multinomialDistribution(int const k)
{
    this->name = std::string("multinomial");
    this->numParams = k;
}

/*!
 * \brief multinomial distribution density function
 * Computes the probability from the multinomial distribution
 * 
 * This function computes the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
 * from a multinomial distribution with probabilities \f$p[K]\f$.
 * 
 * \param  p       Vector of probabilities \f$ p_1, \cdots, p_k \f$ (with size of K)
 * \param  mndist  A random sample (with size of K) from the multinomial distribution 
 * 
 * \returns Density function value (the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$)
 */
template <typename T>
template <unsigned int>
inline T multinomialDistribution<T>::f(T const *p, unsigned int const *mndist)
{
    // compute the total number of independent trials
    unsigned int const N = std::accumulate(mndist, mndist + this->numParams, 0);

    T const totpsum = std::accumulate(p, p + this->numParams, 0);

    T log_pdf = factorial<T>(N);

    for (int i = 0; i < this->numParams; i++)
    {
        if (mndist[i] > 0)
        {
            log_pdf += std::log(p[i] / totpsum) * mndist[i] - factorial<T>(mndist[i]);
        }
    }

    return std::exp(log_pdf);
}

/*!
 * \brief Log of multinomial distribution density function
 * Computes the logarithm of the probability from the multinomial distribution
 * 
 * This function computes the logarithm of the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
 * from a multinomial distribution with probabilities \f$p[K]\f$.
 * 
 * \param p       Vector of probabilities \f$ p_1, \cdots, p_k \f$ (with size of K)
 * \param mndist  A random sample (with size of K) from the multinomial distribution
 * 
 * \returns  Log of density function value (the logarithm of the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ )
 */
template <typename T>
template <unsigned int>
inline T multinomialDistribution<T>::lf(T const *p, unsigned int const *mndist)
{
    // compute the total number of independent trials
    unsigned int N = std::accumulate(mndist, mndist + this->numParams, 0);

    T const totpsum = std::accumulate(p, p + this->numParams, 0);

    // Currently we have the limitation of float or double type in factorial implementation
    T log_pdf = factorial<T>(N);

    for (int i = 0; i < this->numParams; i++)
    {
        if (mndist[i] > 0)
        {
            log_pdf += std::log(p[i] / totpsum) * mndist[i] - factorial<T>(mndist[i]);
        }
    }

    return log_pdf;
}

#endif // UMUQ_MULTINOMIALDISTRIBUTION_H