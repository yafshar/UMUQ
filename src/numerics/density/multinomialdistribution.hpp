#ifndef UMUQ_MULTINOMIALDISTRIBUTION_H
#define UMUQ_MULTINOMIALDISTRIBUTION_H

#include "core/core.hpp"
#include "numerics/function/densityfunction.hpp"
#include "datatype/eigendatatype.hpp"
#include "numerics/eigenlib.hpp"

#include <cmath>

#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <numeric>
#include <functional>

namespace umuq
{

inline namespace density
{

/*! \class multinomialDistribution
 * \ingroup Density_Module
 *
 * \brief The multinomial distribution
 *
 * \tparam RealType     Data type
 * \tparam FunctionType Function type
 *
 * The [multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution) models the
 * probability of counts for rolling a K-sided die n times.
 * For n independent trials each of which leads to a success for exactly one of K categories, with
 * each category having a given fixed success probability, the multinomial distribution gives the
 * probability of any particular combination of numbers of successes for the various categories.
 *
 * Reference:<br>
 * https://en.wikipedia.org/wiki/Multinomial_distribution
 *
 *
 * This class provides probability density \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$
 * from a multinomial distribution with probabilities \f$p[K]\f$. <br>
 * using:<br>
 *
 * \f$
 *     Pr(X_1=n_1, \cdots, X_K=n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \cdots p_K^{n_K}
 * \f$ <br>
 *
 * where \f$ n_1, \cdots n_K \f$ are non-negative integers satisfying \f$ sum_{i=1}^{K} {n_i} = N\f$,
 * and \f$p = \left(p_1, \cdots, p_K\right)\f$ is a probability distribution.
 *
 * This class also provides random integer values x, distributed according to the multinomial distribution probability
 * density function. \sa sample
 */
template <typename RealType, class FunctionType = std::function<RealType(RealType const *, unsigned int const *)>>
class multinomialDistribution : public densityFunction<RealType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new multinomial distribution object
     *
     * \param K  Size of vector which shows K possible mutually exclusive outcomes
     */
    explicit multinomialDistribution(int const K);

    /*!
     * \brief Construct a new multinomialDistribution object with a multinomial distribution \f$ M_K\left(N, p\right) \f$
     *
     * \param p  Vector of probabilities \f$ p_1, \cdots, p_k \f$
     * \param K  Size of vector which shows K possible mutually exclusive outcomes
     * \param N  N independent trials
     */
    multinomialDistribution(RealType const *p, int const K, int const N);

    /*!
     * \brief Destroy the multinomial distribution object
     *
     */
    ~multinomialDistribution();

    /*!
     * \brief Reset the multinomial distribution object
     *
     * \param p  Vector of probabilities \f$ p_1, \cdots, p_k \f$
     * \param K  Size of vector which shows K possible mutually exclusive outcomes
     * \param N  N independent trials
     */
    void reset(RealType const *p, int const K, int const N);

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
    RealType multinomialDistribution_f(RealType const *p, unsigned int const *mndist);

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
    RealType multinomialDistribution_lf(RealType const *p, unsigned int const *mndist);

    /*!
     * \brief Create samples from multinomial distribution
     *
     * \param x  Vector of samples
     */
    inline void sample(int *x);

    /*!
     * \brief Create samples from multinomial distribution
     *
     * \param x  Vector of samples
     */
    inline void sample(std::vector<int> &x);

    /*!
     * \brief Create samples from multinomial distribution
     *
     * \param x  Vector of samples
     */
    inline void sample(EVectorX<int> &x);

    /*!
     * \brief Create samples from multinomial distribution
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     */
    inline void sample(int *x, int const nSamples);

    /*!
     * \brief Create samples from multinomial distribution
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     */
    inline void sample(std::vector<int> &x, int const nSamples);

    /*!
     * \brief Create samples from multinomial distribution
     *
     * \param x  Matrix of random samples
     */
    inline void sample(EMatrixX<int> &x);

  private:
    /*! Multinomial distribution */
    std::unique_ptr<randomdist::multinomialDistribution<RealType>> multinomial;
};

template <typename RealType, class FunctionType>
multinomialDistribution<RealType, FunctionType>::multinomialDistribution(int const K) : multinomial(nullptr)
{
    this->name = std::string("multinomial");
    this->numParams = K;
    this->f = std::bind(&multinomialDistribution<RealType>::multinomialDistribution_f, this, std::placeholders::_1, std::placeholders::_2);
    this->lf = std::bind(&multinomialDistribution<RealType>::multinomialDistribution_lf, this, std::placeholders::_1, std::placeholders::_2);
}

template <typename RealType, class FunctionType>
multinomialDistribution<RealType, FunctionType>::multinomialDistribution(RealType const *p, int const K, int const N) : multinomial(nullptr)
{
    this->name = std::string("multinomial");
    this->numParams = K;
    this->f = std::bind(&multinomialDistribution<RealType>::multinomialDistribution_f, this, std::placeholders::_1, std::placeholders::_2);
    this->lf = std::bind(&multinomialDistribution<RealType>::multinomialDistribution_lf, this, std::placeholders::_1, std::placeholders::_2);
    try
    {
        multinomial.reset(new randomdist::multinomialDistribution<RealType>(p, K, N));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
multinomialDistribution<RealType, FunctionType>::~multinomialDistribution() {}

template <typename RealType, class FunctionType>
void multinomialDistribution<RealType, FunctionType>::reset(RealType const *p, int const K, int const N)
{
    this->numParams = K;
    try
    {
        multinomial.reset(new randomdist::multinomialDistribution<RealType>(p, K, N));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
RealType multinomialDistribution<RealType, FunctionType>::multinomialDistribution_f(RealType const *p, unsigned int const *mndist)
{
    return std::exp(multinomialDistribution_lf(p, mndist));
}

template <typename RealType, class FunctionType>
RealType multinomialDistribution<RealType, FunctionType>::multinomialDistribution_lf(RealType const *p, unsigned int const *mndist)
{
#ifdef DEBUG
    for (int i = 0; i < this->numParams; i++)
    {
        if (p[i] <= RealType{})
        {
            return std::numeric_limits<RealType>::infinity();
        }
    }
#endif
    // compute the total number of independent trials
    unsigned int const N1 = std::accumulate(mndist, mndist + this->numParams, 0) + 1;

    RealType const totpsum = std::accumulate(p, p + this->numParams, RealType{});

    // natural logarithm of the gamma function ~ log(N!)
    RealType log_pdf = std::lgamma(N1);
    for (int i = 0; i < this->numParams; i++)
    {
        if (mndist[i] > 0)
        {
            log_pdf += std::log(p[i] / totpsum) * mndist[i] - std::lgamma(mndist[i] + 1);
        }
    }
    return log_pdf;
}

template <typename RealType, class FunctionType>
inline void multinomialDistribution<RealType, FunctionType>::sample(int *x)
{
    if (multinomial)
    {
        EVectorMapType<int> X(x, this->numParams);
        X = multinomial->dist();
        return;
    }
    UMUQFAIL("The Multinomial distribution object is not assigned! Please reset the object with the right parameters!");
}

template <typename RealType, class FunctionType>
inline void multinomialDistribution<RealType, FunctionType>::sample(std::vector<int> &x)
{
    if (multinomial)
    {
        EVectorMapType<int> X(x.data(), this->numParams);
        X = multinomial->dist();
        return;
    }
    UMUQFAIL("The Multinomial distribution object is not assigned! Please reset the object with the right parameters!");
}

template <typename RealType, class FunctionType>
inline void multinomialDistribution<RealType, FunctionType>::sample(EVectorX<int> &x)
{
    if (multinomial)
    {
        x = multinomial->dist();
        return;
    }
    UMUQFAIL("The Multinomial distribution object is not assigned! Please reset the object with the right parameters!");
}

template <typename RealType, class FunctionType>
inline void multinomialDistribution<RealType, FunctionType>::sample(int *x, int const nSamples)
{
    if (multinomial)
    {
        multinomial->dist(x, this->numParams, nSamples);
        return;
    }
    UMUQFAIL("The Multinomial distribution object is not assigned! Please reset the object with the right parameters!");
}

template <typename RealType, class FunctionType>
inline void multinomialDistribution<RealType, FunctionType>::sample(std::vector<int> &x, int const nSamples)
{
    if (multinomial)
    {
        multinomial->dist(x.data(), this->numParams, nSamples);
        return;
    }
    UMUQFAIL("The Multinomial distribution object is not assigned! Please reset the object with the right parameters!");
}

template <typename RealType, class FunctionType>
inline void multinomialDistribution<RealType, FunctionType>::sample(EMatrixX<int> &x)
{
    if (multinomial)
    {
        multinomial->dist(x);
        return;
    }
    UMUQFAIL("The Multinomial distribution object is not assigned! Please reset the object with the right parameters!");
}

} // namespace density
} // namespace umuq

#endif // UMUQ_MULTINOMIALDISTRIBUTION
