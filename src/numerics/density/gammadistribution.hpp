#ifndef UMUQ_GAMMADISTRIBUTION_H
#define UMUQ_GAMMADISTRIBUTION_H

#include "core/core.hpp"
#include "numerics/function/densityfunction.hpp"
#include "numerics/random/psrandom.hpp"
#include "misc/arraywrapper.hpp"

#include <cstddef>
#include <cmath>

#include <vector>
#include <limits>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>

namespace umuq
{

inline namespace density
{

/*! \class gammaDistribution
 * \ingroup Density_Module
 *
 * \brief The Gamma distribution
 *
 * \tparam RealType     Data type
 * \tparam FunctionType Function type
 *
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a
 * Gamma distribution with shape parameter \f$\alpha > 0\f$ and scale parameter \f$ beta > 0\f$. <br>
 * The scale parameter, \f$ beta\f$, is optional and defaults to \f$ beta = 1\f$. <br>
 * using:<br>
 *
 * \f$
 * p(x)=\frac{1}{\Gamma (\alpha) \beta^\alpha}x^{\alpha-1}e^{\frac{-x}{\beta}}.
 * \f$
 *
 * Use the Gamma distribution with \f$\alpha > 1\f$ if you have a sharp lower bound of zero but no
 * sharp upper bound, a single mode, and a positive skew. The Gamma distribution is especially appropriate
 * when encoding arrival times for sets of events. A Gamma distribution with a large value for \f$\alpha\f$
 * is also useful when you wish to use a bell-shaped curve for a positive-only quantity.
 *
 * This class also provides random non-negative values x, distributed according to the Gamma distribution
 * probability density function. \sa sample
 *
 * \note
 * - \f$ \alpha > 0 \f$
 * - \f$ \beta > 0 \f$
 */

template <typename RealType, class FunctionType = std::function<RealType(RealType const *)>>
class gammaDistribution : public densityFunction<RealType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new Gamma distribution object
     *
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     */
    gammaDistribution(RealType const alpha, RealType const beta = RealType{1});

    /*!
     * \brief Construct a new gamma Distribution object
     *
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     * \param n      Total number of alpha + beta inputs
     */
    gammaDistribution(RealType const *alpha, RealType const *beta, int const n);

    /*!
     * \brief Destroy the Gamma distribution object
     *
     */
    ~gammaDistribution();

    /*!
     * \brief Gamma distribution density function
     *
     * \param x Input value
     *
     * \returns Density function value
     */
    inline RealType gammaDistribution_f(RealType const *x);

    /*!
     * \brief Log of Gamma distribution density function
     *
     * \param x Input value
     *
     * \returns  Log of density function value
     */
    inline RealType gammaDistribution_lf(RealType const *x);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(RealType *x);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(std::vector<RealType> &x);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(EVectorX<RealType> &x);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(RealType *x, int const nSamples);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(std::vector<RealType> &x, int const nSamples);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x  Matrix of random samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(EMatrixX<RealType> &x);

  private:
    /*! \brief Gamma random number distribution of RealType type */
    std::unique_ptr<randomdist::gammaDistribution<RealType>> gamma;

    /*! \brief Gamma random number distributions of RealType type  */
    std::unique_ptr<randomdist::gammaDistribution<RealType>[]> gammas;

    /*! Number of Gamma distributions \sa gammas */
    int ngammas;
};

template <typename RealType, class FunctionType>
gammaDistribution<RealType, FunctionType>::gammaDistribution(RealType const alpha, RealType const beta) : densityFunction<RealType, FunctionType>(&alpha, &beta, 2, "gamma"),
                                                                                                          gamma(nullptr),
                                                                                                          gammas(nullptr),
                                                                                                          ngammas(0)
{
    this->f = std::bind(&gammaDistribution<RealType, FunctionType>::gammaDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gammaDistribution<RealType, FunctionType>::gammaDistribution_lf, this, std::placeholders::_1);
    try
    {
        gamma.reset(new randomdist::gammaDistribution<RealType>(alpha, beta));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
gammaDistribution<RealType, FunctionType>::gammaDistribution(RealType const *alpha, RealType const *beta, int const n) : densityFunction<RealType, FunctionType>(alpha, beta, n, "gamma"),
                                                                                                                         gamma(nullptr),
                                                                                                                         gammas(nullptr),
                                                                                                                         ngammas(n / 2)
{
    if (n & 1)
    {
        UMUQFAIL("Wrong number of inputs!")
    }
    this->f = std::bind(&gammaDistribution<RealType, FunctionType>::gammaDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gammaDistribution<RealType, FunctionType>::gammaDistribution_lf, this, std::placeholders::_1);
    try
    {
        gammas.reset(new randomdist::gammaDistribution<RealType>[ngammas]);
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
    for (int i = 0; i < ngammas; i++)
    {
        gammas[i] = std::move(randomdist::gammaDistribution<RealType>(alpha[i], beta[i]));
    }
}

template <typename RealType, class FunctionType>
gammaDistribution<RealType, FunctionType>::~gammaDistribution() {}

template <typename RealType, class FunctionType>
inline RealType gammaDistribution<RealType, FunctionType>::gammaDistribution_f(RealType const *x)
{
    RealType sum(1);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        if (x[i] < RealType{})
        {
            return RealType{};
        }
        else if (x[i] == RealType{})
        {
            if (this->params[k] == static_cast<RealType>(1))
            {
                sum *= static_cast<RealType>(1) / this->params[k + 1];
                continue;
            }
            else
            {
                return RealType{};
            }
        }
        else if (this->params[k] == static_cast<RealType>(1))
        {
            sum *= std::exp(-x[i] / this->params[k + 1]) / this->params[k + 1];
        }
        else
        {
            sum *= std::exp((this->params[k] - static_cast<RealType>(1)) * std::log(x[i] / this->params[k + 1]) - x[i] / this->params[k + 1] - std::lgamma(this->params[k])) / this->params[k + 1];
        }
    }
    return sum;
}

template <typename RealType, class FunctionType>
inline RealType gammaDistribution<RealType, FunctionType>::gammaDistribution_lf(RealType const *x)
{
    for (std::size_t i = 0; i < this->numParams / 2; i++)
    {
        if (x[i] < RealType{})
        {
            return -std::numeric_limits<RealType>::infinity();
        }
    }
    RealType sum(0);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        sum += -std::lgamma(this->params[k]) - this->params[k] * std::log(this->params[k + 1]) + (this->params[k] - static_cast<RealType>(1)) * std::log(x[i]) - x[i] / this->params[k + 1];
    }
    return sum;
}

template <typename RealType, class FunctionType>
void gammaDistribution<RealType, FunctionType>::sample(RealType *x)
{
    if (gamma)
    {
        x[0] = gamma->dist();
        return;
    }
    for (int i = 0; i < ngammas; i++)
    {
        x[i] = gammas[i].dist();
    }
}

template <typename RealType, class FunctionType>
void gammaDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x)
{
    if (gamma)
    {
        x[0] = gamma->dist();
        return;
    }
    for (int i = 0; i < ngammas; i++)
    {
        x[i] = gammas[i].dist();
    }
}

template <typename RealType, class FunctionType>
void gammaDistribution<RealType, FunctionType>::sample(EVectorX<RealType> &x)
{
    if (gamma)
    {
        x[0] = gamma->dist();
        return;
    }
    for (int i = 0; i < ngammas; i++)
    {
        x[i] = gammas[i].dist();
    }
}

template <typename RealType, class FunctionType>
void gammaDistribution<RealType, FunctionType>::sample(RealType *x, int const nSamples)
{
    if (gamma)
    {
        gamma->dist(x, nSamples);
        return;
    }
    std::size_t const nSizeArray = ngammas * static_cast<std::size_t>(nSamples);
    std::vector<RealType> X(nSamples);
    for (int i = 0; i < ngammas; i++)
    {
        arrayWrapper<RealType> xArray(x + i, nSizeArray, ngammas);
        gammas[i].dist(X);
        std::copy(X.begin(), X.end(), xArray.begin());
    }
}

template <typename RealType, class FunctionType>
void gammaDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x, int const nSamples)
{
    if (gamma)
    {
#ifdef DEBUG
        if (static_cast<std::size_t>(nSamples) > x.size())
        {
            UMUQFAIL("The input array size of ", x.size(), " < requested number of ", nSamples, " samples!");
        }
#endif
        gamma->dist(x);
        return;
    }
    std::size_t const nSizeArray = ngammas * static_cast<std::size_t>(nSamples);
#ifdef DEBUG
    if (nSizeArray > x.size())
    {
        UMUQFAIL("The input array size of ", x.size(), " < requested samples size of ", nSizeArray, " !");
    }
#endif
    std::vector<RealType> X(nSamples);
    for (int i = 0; i < ngammas; i++)
    {
        arrayWrapper<RealType> xArray(x.data() + i, nSizeArray, ngammas);
        gammas[i].dist(X);
        std::copy(X.begin(), X.end(), xArray.begin());
    }
}

template <typename RealType, class FunctionType>
void gammaDistribution<RealType, FunctionType>::sample(EMatrixX<RealType> &x)
{
#ifdef DEBUG
    if (this->numParams / 2 != x.rows())
    {
        UMUQFAIL("The input dimension =", x.rows(), " != samples dimension of ", this->numParams / 2, " !");
    }
#endif
    std::vector<RealType> X(x.cols());
    if (gamma)
    {
        gamma->dist(X);
        for (auto j = 0; j < x.cols(); ++j)
        {
            x(0, j) = X[j];
        }
        return;
    }
    for (int i = 0; i < ngammas; ++i)
    {
        gammas[i].dist(X);
        for (auto j = 0; j < x.cols(); ++j)
        {
            x(i, j) = X[j];
        }
    }
}

} // namespace density
} // namespace umuq

#endif // UMUQ_GAMMADISTRIBUTION
