#ifndef UMUQ_EXPONENTIALDISTRIBUTION_H
#define UMUQ_EXPONENTIALDISTRIBUTION_H

#include "core/core.hpp"
#include "numerics/function/densityfunction.hpp"

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

/*! \class exponentialDistribution
 * \ingroup Density_Module
 *
 * \brief The exponential distribution
 *
 * \tparam RealType     Data type
 * \tparam FunctionType Function type
 *
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for an
 * exponential distribution of:
 *
 * \f$
 * p(x)=\frac{1}{\mu}e^{\left(-\frac{x}{\mu}\right)},
 * \f$<br>
 *
 * where \f$ \mu > 0 \f$ is mean, standard deviation, and scale parameter of the
 * distribution, the reciprocal of the rate parameter in an another commonly used
 * alternative parametrization of:<br>
 *
 * \f$
 * p(x)=\lambda e^{\left(-\lambda x\right)},
 * \f$<br>
 *
 * where \f$ \lambda > 0 \f$ is rate.
 *
 * This class also provides random non-negative values x, distributed according to the exponential
 * distribution probability density function. \sa sample
 *
 * \note
 * - Requires that \f$ \mu > 0 \f$.
 */
template <typename RealType, class FunctionType = std::function<RealType(RealType const *)>>
class exponentialDistribution : public densityFunction<RealType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new exponential distribution object
     *
     * \param mu Mean, \f$ \mu \f$
     */
    explicit exponentialDistribution(RealType const mu);

    /*!
     * \brief Construct a new exponential Distribution object
     *
     * \param mu Mean, \f$ \mu \f$
     * \param n  Number of input
     */
    explicit exponentialDistribution(RealType const *mu, int const n);

    /*!
     * \brief Destroy the exponential distribution object
     *
     */
    ~exponentialDistribution();

    /*!
     * \brief Exponential distribution density function
     *
     * \param x Input value
     *
     * \returns Density function value
     */
    inline RealType exponentialDistribution_f(RealType const *x);

    /*!
     * \brief Log of exponential distribution density function
     *
     * \param x Input value
     *
     * \returns  Log of density function value
     */
    inline RealType exponentialDistribution_lf(RealType const *x);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(RealType *x);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(std::vector<RealType> &x);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(EVectorX<RealType> &x);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(RealType *x, int const nSamples);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(std::vector<RealType> &x, int const nSamples);

    /*!
     * \brief  Create samples of the exponential distribution object
     *
     * \param x  Matrix of random samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(EMatrixX<RealType> &x);

  private:
    /*!
     * \brief Delete an empty exponentialDistribution object construction
     */
    exponentialDistribution() = delete;

  private:
    /*! Exponential random number distribution of RealType type */
    std::unique_ptr<randomdist::exponentialDistribution<RealType>> expn;

    /*! Exponential random number distributions of RealType type */
    std::unique_ptr<randomdist::exponentialDistribution<RealType>[]> expns;

    /*! Number of Exponential distributions \sa expns */
    int nexpns;
};

template <typename RealType, class FunctionType>
exponentialDistribution<RealType, FunctionType>::exponentialDistribution(RealType const mu) : densityFunction<RealType, FunctionType>(&mu, 1, "exponential"),
                                                                                              expn(nullptr),
                                                                                              expns(nullptr),
                                                                                              nexpns(0)
{
    this->f = std::bind(&exponentialDistribution<RealType, FunctionType>::exponentialDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&exponentialDistribution<RealType, FunctionType>::exponentialDistribution_lf, this, std::placeholders::_1);
    try
    {
        expn.reset(new randomdist::exponentialDistribution<RealType>(mu));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
exponentialDistribution<RealType, FunctionType>::exponentialDistribution(RealType const *mu, int const n) : densityFunction<RealType, FunctionType>(mu, n, "exponential"),
                                                                                                            expn(nullptr),
                                                                                                            expns(nullptr),
                                                                                                            nexpns(n)
{
    this->f = std::bind(&exponentialDistribution<RealType, FunctionType>::exponentialDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&exponentialDistribution<RealType, FunctionType>::exponentialDistribution_lf, this, std::placeholders::_1);
    try
    {
        expns.reset(new randomdist::exponentialDistribution<RealType>[nexpns]);
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
    for (int i = 0; i < nexpns; i++)
    {
        expns[i] = std::move(randomdist::exponentialDistribution<RealType>(mu[i]));
    }
}

template <typename RealType, class FunctionType>
exponentialDistribution<RealType, FunctionType>::~exponentialDistribution() {}

template <typename RealType, class FunctionType>
inline RealType exponentialDistribution<RealType, FunctionType>::exponentialDistribution_f(RealType const *x)
{
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        if (x[i] < RealType{})
        {
            return RealType{};
        }
    }
    RealType sum(1);
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        sum *= std::exp(-x[i] / this->params[i]) / this->params[i];
    }
    return sum;
}

template <typename RealType, class FunctionType>
inline RealType exponentialDistribution<RealType, FunctionType>::exponentialDistribution_lf(RealType const *x)
{
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        if (x[i] < RealType{})
        {
            return std::numeric_limits<RealType>::infinity();
        }
    }
    RealType sum(0);
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        sum -= (std::log(this->params[i]) + x[i] / this->params[i]);
    }
    return sum;
}

template <typename RealType, class FunctionType>
void exponentialDistribution<RealType, FunctionType>::sample(RealType *x)
{
    if (expn)
    {
        x[0] = expn->dist();
        return;
    }
    for (int i = 0; i < nexpns; i++)
    {
        x[i] = expns[i].dist();
    }
}

template <typename RealType, class FunctionType>
void exponentialDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x)
{
    if (expn)
    {
        x[0] = expn->dist();
        return;
    }
    for (int i = 0; i < nexpns; i++)
    {
        x[i] = expns[i].dist();
    }
}

template <typename RealType, class FunctionType>
void exponentialDistribution<RealType, FunctionType>::sample(EVectorX<RealType> &x)
{
    if (expn)
    {
        x[0] = expn->dist();
        return;
    }
    for (int i = 0; i < nexpns; i++)
    {
        x[i] = expns[i].dist();
    }
}

template <typename RealType, class FunctionType>
void exponentialDistribution<RealType, FunctionType>::sample(RealType *x, int const nSamples)
{
    if (expn)
    {
        expn->dist(x, nSamples);
        return;
    }
    std::size_t const nSizeArray = nexpns * static_cast<std::size_t>(nSamples);
    std::vector<RealType> X(nSamples);
    for (int i = 0; i < nexpns; i++)
    {
        arrayWrapper<RealType> xArray(x + i, nSizeArray, nexpns);
        expns[i].dist(X);
        std::copy(X.begin(), X.end(), xArray.begin());
    }
}

template <typename RealType, class FunctionType>
void exponentialDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x, int const nSamples)
{
    if (expn)
    {
#ifdef DEBUG
        if (static_cast<std::size_t>(nSamples) > x.size())
        {
            UMUQFAIL("The input array size of ", x.size(), " < requested number of ", nSamples, " samples!");
        }
#endif
        expn->dist(x);
        return;
    }
    std::size_t const nSizeArray = nexpns * static_cast<std::size_t>(nSamples);
#ifdef DEBUG
    if (nSizeArray > x.size())
    {
        UMUQFAIL("The input array size of ", x.size(), " < requested samples size of ", nSizeArray, " !");
    }
#endif
    std::vector<RealType> X(nSamples);
    for (int i = 0; i < nexpns; i++)
    {
        arrayWrapper<RealType> xArray(x.data() + i, nSizeArray, nexpns);
        expns[i].dist(X);
        std::copy(X.begin(), X.end(), xArray.begin());
    }
}

template <typename RealType, class FunctionType>
void exponentialDistribution<RealType, FunctionType>::sample(EMatrixX<RealType> &x)
{
#ifdef DEBUG
    if (this->numParams != x.rows())
    {
        UMUQFAIL("The input dimension =", x.rows(), " != samples dimension of ", this->numParams, " !");
    }
#endif
    std::vector<RealType> X(x.cols());
    if (expn)
    {
        expn->dist(X);
        for (auto j = 0; j < x.cols(); ++j)
        {
            x(0, j) = X[j];
        }
        return;
    }
    for (int i = 0; i < nexpns; ++i)
    {
        expns[i].dist(X);
        for (auto j = 0; j < x.cols(); ++j)
        {
            x(i, j) = X[j];
        }
    }
}

} // namespace density
} // namespace umuq

#endif //UMUQ_EXPONENTIALDISTRIBUTION
