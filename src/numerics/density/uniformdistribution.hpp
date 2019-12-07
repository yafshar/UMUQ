#ifndef UMUQ_UNIFORMDISTRIBUTION_H
#define UMUQ_UNIFORMDISTRIBUTION_H

#include "core/core.hpp"
#include "numerics/function/densityfunction.hpp"
#include "numerics/random/psrandom.hpp"
#include "misc/arraywrapper.hpp"

#include <cstddef>
#include <cmath>

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <limits>
#include <functional>

namespace umuq
{

inline namespace density
{

/*! \class uniformDistribution
 * \ingroup Density_Module
 *
 * \brief Flat (Uniform) distribution function
 *
 * \tparam RealType     Data type
 * \tparam FunctionType Function type
 *
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a uniform distribution
 * on the closed interval \f$ [low \cdots high] \f$, <br>
 * using:<br>
 *
 * \f$
 * p(x)= \left\{
 * \begin{matrix}
 * 1/(high-low)  &low \leq x < high \\
 *  0       &otherwise
 * \end{matrix}
 * \right.
 * \f$
 *
 * This class also provides sample of random values x, distributed according to the uniform distribution
 * probability density function. \sa sample
 */
template <typename RealType, class FunctionType = std::function<RealType(RealType const *)>>
class uniformDistribution : public densityFunction<RealType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new uniform Distribution object
     *
     * \param low  Lower bound
     * \param high  Upper bound
     */
    uniformDistribution(RealType const low, RealType const high);

    /*!
     * \brief Construct a new uniform Distribution object
     *
     * \param low   Lower bound
     * \param high  Upper bound
     * \param n     Total number of Lower bound + Upper bound inputs
     */
    uniformDistribution(RealType const *low, RealType const *high, int const n);

    /*!
     * \brief Destroy the uniform Distribution object
     *
     */
    ~uniformDistribution();

    /*!
     * \brief Uniform distribution density function
     *
     * \param x  Input value
     *
     * \returns  Density function value
     */
    inline RealType uniformDistribution_f(RealType const *x);

    /*!
     * \brief Log of Uniform distribution density function
     *
     * \param x  Input value
     *
     * \returns  Log of density function value
     */
    inline RealType uniformDistribution_lf(RealType const *x);

    /*!
     * \brief Create samples of the uniform Distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(RealType *x);

    /*!
     * \brief Create samples of the uniform Distribution object
     *
     * \param x  Vector of samples
     */
    void sample(std::vector<RealType> &x);

    /*!
     * \brief Create samples of the uniform Distribution object
     *
     * \param x  Vector of samples
     */
    void sample(EVectorX<RealType> &x);

    /*!
     * \brief Create samples of the uniform Distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     */
    void sample(RealType *x, int const nSamples);

    /*!
     * \brief Create samples of the uniform Distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     */
    void sample(std::vector<RealType> &x, int const nSamples);

    /*!
     * \brief Create samples of the uniform Distribution object
     *
     * \param x  Matrix of random samples
     */
    void sample(EMatrixX<RealType> &x);

  private:
    /*! Const value for uniform distribution function */
    RealType uniformDistribution_fValue;

    /*! Const value for logarithm of the uniform distribution function */
    RealType uniformDistribution_lfValue;

    /*! Helper function */
    inline RealType uniformDistribution_f_();

    /*! Helper log function */
    inline RealType uniformDistribution_lf_();

    /*! Uniform random number distribution of RealType */
    std::unique_ptr<randomdist::uniformDistribution<RealType>> uniform;

    /*! Uniform random number distribution of RealType type */
    std::unique_ptr<randomdist::uniformDistribution<RealType>[]> uniforms;

    /*! Number of uniform distributions. \sa uniforms */
    int nuniforms;
};

template <typename RealType, class FunctionType>
uniformDistribution<RealType, FunctionType>::uniformDistribution(RealType const low, RealType const high) : densityFunction<RealType, FunctionType>(&low, &high, 2, "uniform"),
                                                                                                            uniform(nullptr),
                                                                                                            uniforms(nullptr),
                                                                                                            nuniforms(0)
{
    this->f = std::bind(&uniformDistribution<RealType>::uniformDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<RealType>::uniformDistribution_lf, this, std::placeholders::_1);
    uniformDistribution_fValue = uniformDistribution_f_();
    uniformDistribution_lfValue = uniformDistribution_lf_();
    try
    {
        uniform.reset(new randomdist::uniformDistribution<RealType>(low, high));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
uniformDistribution<RealType, FunctionType>::uniformDistribution(RealType const *low, RealType const *high, int const n) : densityFunction<RealType, FunctionType>(low, high, n, "uniform"),
                                                                                                                           uniform(nullptr),
                                                                                                                           uniforms(nullptr),
                                                                                                                           nuniforms(n / 2)
{
    if (n & 1)
    {
        UMUQFAIL("Wrong number of inputs!");
    }
    this->f = std::bind(&uniformDistribution<RealType>::uniformDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<RealType>::uniformDistribution_lf, this, std::placeholders::_1);
    uniformDistribution_fValue = uniformDistribution_f_();
    uniformDistribution_lfValue = uniformDistribution_lf_();
    try
    {
        uniforms.reset(new randomdist::uniformDistribution<RealType>[nuniforms]);
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
    for (int i = 0; i < nuniforms; i++)
    {
        uniforms[i] = std::move(randomdist::uniformDistribution<RealType>(low[i], high[i]));
    }
}

template <typename RealType, class FunctionType>
uniformDistribution<RealType, FunctionType>::~uniformDistribution() {}

template <typename RealType, class FunctionType>
inline RealType uniformDistribution<RealType, FunctionType>::uniformDistribution_f(RealType const *x)
{
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        if (x[i] < this->params[k] || x[i] >= this->params[k + 1])
        {
            return RealType{};
        }
    }
    return uniformDistribution_fValue;
}

template <typename RealType, class FunctionType>
inline RealType uniformDistribution<RealType, FunctionType>::uniformDistribution_lf(RealType const *x)
{
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        if (x[i] < this->params[k] || x[i] >= this->params[k + 1])
        {
            return std::numeric_limits<RealType>::infinity();
        }
    }
    return uniformDistribution_lfValue;
}

template <typename RealType, class FunctionType>
inline RealType uniformDistribution<RealType, FunctionType>::uniformDistribution_f_()
{
    RealType sum(1);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        sum *= static_cast<RealType>(1) / (this->params[k + 1] - this->params[k]);
    }
    return sum;
}

template <typename RealType, class FunctionType>
inline RealType uniformDistribution<RealType, FunctionType>::uniformDistribution_lf_()
{
    RealType sum(0);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        sum -= std::log(this->params[k + 1] - this->params[k]);
    }
    return sum;
}

template <typename RealType, class FunctionType>
void uniformDistribution<RealType, FunctionType>::sample(RealType *x)
{
    if (uniform)
    {
        x[0] = uniform->dist();
        return;
    }
    for (int i = 0; i < nuniforms; i++)
    {
        x[i] = uniforms[i].dist();
    }
}

template <typename RealType, class FunctionType>
void uniformDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x)
{
    if (uniform)
    {
        x[0] = uniform->dist();
        return;
    }
    for (int i = 0; i < nuniforms; i++)
    {
        x[i] = uniforms[i].dist();
    }
}

template <typename RealType, class FunctionType>
void uniformDistribution<RealType, FunctionType>::sample(EVectorX<RealType> &x)
{
    if (uniform)
    {
        x[0] = uniform->dist();
        return;
    }
    for (int i = 0; i < nuniforms; i++)
    {
        x[i] = uniforms[i].dist();
    }
}

template <typename RealType, class FunctionType>
void uniformDistribution<RealType, FunctionType>::sample(RealType *x, int const nSamples)
{
    if (uniform)
    {
        uniform->dist(x, nSamples);
        return;
    }
    std::size_t const nSizeArray = nuniforms * static_cast<std::size_t>(nSamples);
    std::vector<RealType> X(nSamples);
    for (int i = 0; i < nuniforms; i++)
    {
        arrayWrapper<RealType> xArray(x + i, nSizeArray, nuniforms);
        uniforms[i].dist(X);
        std::copy(X.begin(), X.end(), xArray.begin());
    }
}

template <typename RealType, class FunctionType>
void uniformDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x, int const nSamples)
{
    if (uniform)
    {
#ifdef DEBUG
        if (static_cast<std::size_t>(nSamples) > x.size())
        {
            UMUQFAIL("The input array size of ", x.size(), " < requested number of ", nSamples, " samples!");
        }
#endif
        uniform->dist(x);
        return;
    }
    std::size_t const nSizeArray = nuniforms * static_cast<std::size_t>(nSamples);
#ifdef DEBUG
    if (nSizeArray > x.size())
    {
        UMUQFAIL("The input array size of ", x.size(), " < requested samples size of ", nSizeArray, " !");
    }
#endif
    std::vector<RealType> X(nSamples);
    for (int i = 0; i < nuniforms; i++)
    {
        uniforms[i].dist(X);
        arrayWrapper<RealType> xArray(x.data() + i, nSizeArray, nuniforms);
        std::copy(X.begin(), X.end(), xArray.begin());
    }
}

template <typename RealType, class FunctionType>
void uniformDistribution<RealType, FunctionType>::sample(EMatrixX<RealType> &x)
{
#ifdef DEBUG
    if (this->numParams / 2 != x.rows())
    {
        UMUQFAIL("The input dimension =", x.rows(), " != samples dimension of ", this->numParams / 2, " !");
    }
#endif
    std::vector<RealType> X(x.cols());
    if (uniform)
    {
        uniform->dist(X);
        for (auto j = 0; j < x.cols(); ++j)
        {
            x(0, j) = X[j];
        }
        return;
    }
    for (int i = 0; i < nuniforms; ++i)
    {
        uniforms[i].dist(X);
        for (auto j = 0; j < x.cols(); ++j)
        {
            x(i, j) = X[j];
        }
    }
}

} // namespace density
} // namespace umuq

#endif //UMUQ_UNIFORMDISTRIBUTION_H
