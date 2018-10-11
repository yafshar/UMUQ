#ifndef UMUQ_MULTINOMIALDISTRIBUTION_H
#define UMUQ_MULTINOMIALDISTRIBUTION_H

namespace umuq
{

inline namespace density
{

/*! \class multinomialDistribution
 * \ingroup Density_Module
 * 
 * \brief The multinomial distribution
 * The multinomial distribution models the probability of counts for rolling a k-sided die n times. 
 * For n independent trials each of which leads to a success for exactly one of k categories, with 
 * each category having a given fixed success probability, the multinomial distribution gives the 
 * probability of any particular combination of numbers of successes for the various categories.
 * 
 * Reference:<br>
 * https://en.wikipedia.org/wiki/Multinomial_distribution
 * 
 * 
 * This class provides probability density \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
 * from a multinomial distribution with probabilities \f$p[K]\f$. <br>
 * using:
 * 
 * \f$
 *     Pr(X_1=n_1, \cdots, X_K=n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \cdots p_K^{n_K}
 * \f$ <br>
 * 
 * where \f$ n_1, \cdots n_K \f$ are nonnegative integers satisfying \f$ sum_{i=1}^{K} {n_i} = N\f$, 
 * and \f$p = \left(p_1, \cdots, p_K\right)\f$ is a probability distribution. 
 * 
 * \tparam T Data type
 */
template <typename T>
class multinomialDistribution : public densityFunction<T, std::function<T(T const *, unsigned int const *)>>
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
    T multinomialDistribution_f(T const *p, unsigned int const *mndist);

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
    T multinomialDistribution_lf(T const *p, unsigned int const *mndist);
};

template <typename T>
multinomialDistribution<T>::multinomialDistribution(int const k)
{
    this->name = std::string("multinomial");
    this->numParams = k;
    this->f = std::bind(&multinomialDistribution<T>::multinomialDistribution_f, this, std::placeholders::_1, std::placeholders::_2);
    this->lf = std::bind(&multinomialDistribution<T>::multinomialDistribution_lf, this, std::placeholders::_1, std::placeholders::_2);
}

template <typename T>
T multinomialDistribution<T>::multinomialDistribution_f(T const *p, unsigned int const *mndist)
{
    return std::exp(multinomialDistribution_lf(p, mndist));
}

template <typename T>
T multinomialDistribution<T>::multinomialDistribution_lf(T const *p, unsigned int const *mndist)
{
#ifdef DEBUG
    for (int i = 0; i < this->numParams; i++)
    {
        if (p[i] <= T{})
        {
            return std::numeric_limits<T>::infinity();
        }
    }
#endif
    // compute the total number of independent trials
    unsigned int const N1 = std::accumulate(mndist, mndist + this->numParams, 0) + 1;

    T const totpsum = std::accumulate(p, p + this->numParams, T{});

    // natural logarithm of the gamma function ~ log(N!)
    T log_pdf = std::lgamma(N1);
    for (int i = 0; i < this->numParams; i++)
    {
        if (mndist[i] > 0)
        {
            log_pdf += std::log(p[i] / totpsum) * mndist[i] - std::lgamma(mndist[i] + 1);
        }
    }
    return log_pdf;
}

} // namespace density
} // namespace umuq

#endif // UMUQ_MULTINOMIALDISTRIBUTION
