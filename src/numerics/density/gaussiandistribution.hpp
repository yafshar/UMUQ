#ifndef UMUQ_GAUSSIANDISTRIBUTION_H
#define UMUQ_GAUSSIANDISTRIBUTION_H

#include "core/core.hpp"
#include "numerics/function/densityfunction.hpp"
#include "numerics/random/psrandom.hpp"
#include "misc/arraywrapper.hpp"

#include <cstddef>
#include <cmath>

#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>

namespace umuq
{

inline namespace density
{

/*! \class gaussianDistribution
 * \ingroup Density_Module
 *
 * \brief The Gaussian distribution
 *
 * \tparam RealType     Data type
 * \tparam FunctionType Function type
 *
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a Gaussian
 * distribution with standard deviation \f$ \sigma \f$ <br>
 * using:<br>
 *
 * \f$
 * p(x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(-\frac{\left(x - \mu \right)^2}{2\sigma^2}\right)}.
 * \f$
 *
 * This class also provides random values x, distributed according to the Gaussian distribution probability
 * density function. \sa sample
 */
template <typename RealType, class FunctionType = std::function<RealType(RealType const *)>>
class gaussianDistribution : public densityFunction<RealType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new gaussian Distribution object
     *
     * \param mu     Mean, \f$ \mu \f$
     * \param sigma  Standard deviation \f$ \sigma \f$
     */
    gaussianDistribution(RealType const mu, RealType const sigma);

    /*!
     * \brief Construct a new gaussian Distribution object
     *
     * \param mu     Mean, \f$ \mu \f$
     * \param sigma  Standard deviation \f$ \sigma \f$
     * \param n      Total number of Mean + Standard deviation inputs
     */
    gaussianDistribution(RealType const *mu, RealType const *sigma, int const n);

    /*!
     * \brief Destroy the gaussian Distribution object
     */
    ~gaussianDistribution();

    /*!
     * \brief Gaussian Distribution density function
     *
     * \param x  Input value
     *
     * \returns Density function value
     */
    inline RealType gaussianDistribution_f(RealType const *x);

    /*!
     * \brief Log of Gaussian Distribution density function
     *
     * \param x  Input value
     *
     * \returns  Log of density function value
     */
    inline RealType gaussianDistribution_lf(RealType const *x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(RealType *x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(std::vector<RealType> &x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(EVectorX<RealType> &x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(RealType *x, int const nSamples);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(std::vector<RealType> &x, int const nSamples);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Matrix of random samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    void sample(EMatrixX<RealType> &x);

  private:
    /*! Normal (or Gaussian) random number distribution of double type */
    std::unique_ptr<randomdist::normalDistribution<RealType>> normal;

    /*! Normals (or Gaussian) random number distribution of double type */
    std::unique_ptr<randomdist::normalDistribution<RealType>[]> normals;

    /*! Number of normal distributions. \sa normals */
    int nnormals;
};

template <typename RealType, class FunctionType>
gaussianDistribution<RealType, FunctionType>::gaussianDistribution(RealType const mu, RealType const sigma) : densityFunction<RealType, FunctionType>(&mu, &sigma, 2, "gaussian"),
                                                                                                              normal(nullptr),
                                                                                                              normals(nullptr),
                                                                                                              nnormals(0)
{
    this->f = std::bind(&gaussianDistribution<RealType, FunctionType>::gaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gaussianDistribution<RealType, FunctionType>::gaussianDistribution_lf, this, std::placeholders::_1);
    try
    {
        normal.reset(new randomdist::normalDistribution<RealType>(mu, sigma));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
gaussianDistribution<RealType, FunctionType>::gaussianDistribution(RealType const *mu, RealType const *sigma, int const n) : densityFunction<RealType, FunctionType>(mu, sigma, n, "gaussian"),
                                                                                                                             normal(nullptr),
                                                                                                                             normals(nullptr),
                                                                                                                             nnormals(n / 2)
{
    if (n & 1)
    {
        UMUQFAIL("Wrong number of inputs!");
    }
    this->f = std::bind(&gaussianDistribution<RealType, FunctionType>::gaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gaussianDistribution<RealType, FunctionType>::gaussianDistribution_lf, this, std::placeholders::_1);
    try
    {
        normals.reset(new randomdist::normalDistribution<RealType>[nnormals]);
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
    for (int i = 0; i < nnormals; i++)
    {
        normals[i] = std::move(randomdist::normalDistribution<RealType>(mu[i], sigma[i]));
    }
}

template <typename RealType, class FunctionType>
gaussianDistribution<RealType, FunctionType>::~gaussianDistribution() {}

template <typename RealType, class FunctionType>
inline RealType gaussianDistribution<RealType, FunctionType>::gaussianDistribution_f(RealType const *x)
{
    RealType sum(1);
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        RealType const xSigma = (x[i] - this->params[k]) / this->params[k + 1];
        sum *= static_cast<RealType>(1) / (M_S2PI * this->params[k + 1]) * std::exp(-0.5 * xSigma * xSigma);
    }
    return sum;
}

template <typename RealType, class FunctionType>
inline RealType gaussianDistribution<RealType, FunctionType>::gaussianDistribution_lf(RealType const *x)
{
    RealType sum(0);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        RealType const xSigma = (x[i] - this->params[k]) / this->params[k + 1];
        sum += -0.5 * M_L2PI - std::log(this->params[k + 1]) - 0.5 * xSigma * xSigma;
    }
    return sum;
}

template <typename RealType, class FunctionType>
void gaussianDistribution<RealType, FunctionType>::sample(RealType *x)
{
    if (normal)
    {
        *x = normal->dist();
        return;
    }
    for (int i = 0; i < nnormals; i++)
    {
        x[i] = normals[i].dist();
    }
}

template <typename RealType, class FunctionType>
void gaussianDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x)
{
    if (normal)
    {
        x[0] = normal->dist();
        return;
    }
    for (int i = 0; i < nnormals; i++)
    {
        x[i] = normals[i].dist();
    }
}

template <typename RealType, class FunctionType>
void gaussianDistribution<RealType, FunctionType>::sample(EVectorX<RealType> &x)
{
    if (normal)
    {
        x[0] = normal->dist();
        return;
    }
    for (int i = 0; i < nnormals; i++)
    {
        x[i] = normals[i].dist();
    }
}

template <typename RealType, class FunctionType>
void gaussianDistribution<RealType, FunctionType>::sample(RealType *x, int const nSamples)
{
    if (normal)
    {
        normal->dist(x, nSamples);
        return;
    }
    std::size_t const nSizeArray = nnormals * static_cast<std::size_t>(nSamples);
    std::vector<RealType> X(nSamples);
    for (int i = 0; i < nnormals; i++)
    {
        arrayWrapper<RealType> xArray(x + i, nSizeArray, nnormals);
        normals[i].dist(X);
        std::copy(X.begin(), X.end(), xArray.begin());
    }
}

template <typename RealType, class FunctionType>
void gaussianDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x, int const nSamples)
{
    if (normal)
    {
#ifdef DEBUG
        if (static_cast<std::size_t>(nSamples) > x.size())
        {
            UMUQFAIL("The input array size of ", x.size(), " < requested number of ", nSamples, " samples!");
        }
#endif
        normal->dist(x);
        return;
    }
    std::size_t const nSizeArray = nnormals * static_cast<std::size_t>(nSamples);
#ifdef DEBUG
    if (nSizeArray > x.size())
    {
        UMUQFAIL("The input array size of ", x.size(), " < requested samples size of ", nSizeArray, " !");
    }
#endif
    std::vector<RealType> X(nSamples);
    for (int i = 0; i < nnormals; i++)
    {
        normals[i].dist(X);
        arrayWrapper<RealType> xArray(x.data() + i, nSizeArray, nnormals);
        std::copy(X.begin(), X.end(), xArray.begin());
    }
}

template <typename RealType, class FunctionType>
void gaussianDistribution<RealType, FunctionType>::sample(EMatrixX<RealType> &x)
{
#ifdef DEBUG
    if (this->numParams / 2 != x.rows())
    {
        UMUQFAIL("The input dimension =", x.rows(), " != samples dimension of ", this->numParams / 2, " !");
    }
#endif
    std::vector<RealType> X(x.cols());
    if (normal)
    {
        normal->dist(X);
        for (auto j = 0; j < x.cols(); ++j)
        {
            x(0, j) = X[j];
        }
        return;
    }
    for (int i = 0; i < nnormals; ++i)
    {
        normals[i].dist(X);
        for (auto j = 0; j < x.cols(); ++j)
        {
            x(i, j) = X[j];
        }
    }
}

} // namespace density
} // namespace umuq

#endif //UMUQ_GAUSSIANDISTRIBUTION
