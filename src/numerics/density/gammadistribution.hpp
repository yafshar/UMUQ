#ifndef UMUQ_GAMMADISTRIBUTION_H
#define UMUQ_GAMMADISTRIBUTION_H

namespace umuq
{

inline namespace density
{

/*! \class gammaDistribution
 * \ingroup Density_Module
 * 
 * \brief The Gamma distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a 
 * Gamma distribution with shape parameter \f$\alpha > 0\f$ and scale parameter \f$ beta > 0\f$. <br>
 * The scale parameter, \f$ beta\f$, is optional and defaults to \f$ beta = 1\f$. <br>
 * using:
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
 * It also provides random non-negative values x, distributed according to the Gamma distribution 
 * probability density function. 
 * 
 * \note
 * - For using sample member function, setting the the Random Number Generator is required, otherwise, it fails.
 * - \f$ \alpha > 0 \f$
 * - \f$ \beta > 0 \f$
 * 
 * \tparam DataType Data type
 */

template <typename DataType, class FunctionType = std::function<DataType(DataType const *)>>
class gammaDistribution : public densityFunction<DataType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new Gamma distribution object
     * 
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     */
    gammaDistribution(DataType const alpha, DataType const beta = DataType{1});

    /*!
     * \brief Construct a new gamma Distribution object
     * 
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     * \param n      Total number of alpha + beta inputs
     */
    gammaDistribution(DataType const *alpha, DataType const *beta, int const n);

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
    inline DataType gammaDistribution_f(DataType const *x);

    /*!
     * \brief Log of Gamma distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline DataType gammaDistribution_lf(DataType const *x);

    /*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
     * 
     * \return false If it encounters an unexpected problem
     */
    inline bool setRandomGenerator(psrandom<DataType> *PRNG);

    /*!
     * \brief Get the Random Number Generator object 
     * 
     * \returns Pseudo-random number object. \sa umuq::random::psrandom.
     */
    inline psrandom<DataType> *getRandomGenerator();

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(DataType *x);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<DataType> &x);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EVectorX<DataType> &x);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(DataType *x, int const nSamples);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<DataType> &x, int const nSamples);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x  Matrix of random samples 
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EMatrixX<DataType> &x);
};

template <typename DataType, class FunctionType>
gammaDistribution<DataType, FunctionType>::gammaDistribution(DataType const alpha, DataType const beta) : densityFunction<DataType, FunctionType>(&alpha, &beta, 2, "gamma")
{
    this->f = std::bind(&gammaDistribution<DataType, FunctionType>::gammaDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gammaDistribution<DataType, FunctionType>::gammaDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
gammaDistribution<DataType, FunctionType>::gammaDistribution(DataType const *alpha, DataType const *beta, int const n) : densityFunction<DataType, FunctionType>(alpha, beta, n, "gamma")
{
    if (n & 1)
    {
        UMUQFAIL("Wrong number of inputs!")
    }
    this->f = std::bind(&gammaDistribution<DataType, FunctionType>::gammaDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gammaDistribution<DataType, FunctionType>::gammaDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
inline DataType gammaDistribution<DataType, FunctionType>::gammaDistribution_f(DataType const *x)
{
    DataType sum(1);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        if (x[i] < DataType{})
        {
            return DataType{};
        }
        else if (x[i] == DataType{})
        {
            if (this->params[k] == static_cast<DataType>(1))
            {
                sum *= static_cast<DataType>(1) / this->params[k + 1];
                continue;
            }
            else
            {
                return DataType{};
            }
        }
        else if (this->params[k] == static_cast<DataType>(1))
        {
            sum *= std::exp(-x[i] / this->params[k + 1]) / this->params[k + 1];
        }
        else
        {
            sum *= std::exp((this->params[k] - static_cast<DataType>(1)) * std::log(x[i] / this->params[k + 1]) - x[i] / this->params[k + 1] - std::lgamma(this->params[k])) / this->params[k + 1];
        }
    }
    return sum;
}

template <typename DataType, class FunctionType>
inline DataType gammaDistribution<DataType, FunctionType>::gammaDistribution_lf(DataType const *x)
{
    for (std::size_t i = 0; i < this->numParams / 2; i++)
    {
        if (x[i] < DataType{})
        {
            return -std::numeric_limits<DataType>::infinity();
        }
    }
    DataType sum(0);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        sum += -std::lgamma(this->params[k]) - this->params[k] * std::log(this->params[k + 1]) + (this->params[k] - static_cast<DataType>(1)) * std::log(x[i]) - x[i] / this->params[k + 1];
    }
    return sum;
}

template <typename DataType, class FunctionType>
inline bool gammaDistribution<DataType, FunctionType>::setRandomGenerator(psrandom<DataType> *PRNG)
{
    if (PRNG)
    {
        if (PRNG_initialized)
        {
            this->prng = PRNG;
            if (this->numParams > 2)
            {
                return this->prng->set_gammas(this->params.data(), this->numParams);
            }
            return this->prng->set_gamma(this->params[0], this->params[1]);
        }
        UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to this distribution!");
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename DataType, class FunctionType>
inline psrandom<DataType> *gammaDistribution<DataType, FunctionType>::getRandomGenerator() { return this->prng; }

template <typename DataType, class FunctionType>
bool gammaDistribution<DataType, FunctionType>::sample(DataType *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (std::size_t i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->gammas[i].dist();
            }
            return true;
        }
        *x = this->prng->gamma->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gammaDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (std::size_t i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->gammas[i].dist();
            }
            return true;
        }
        x[0] = this->prng->gamma->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gammaDistribution<DataType, FunctionType>::sample(EVectorX<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (std::size_t i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->gammas[i].dist();
            }
            return true;
        }
        x[0] = this->prng->gamma->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gammaDistribution<DataType, FunctionType>::sample(DataType *x, int const nSamples)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            std::size_t const nDim = this->numParams / 2;
            for (std::size_t j = 0, l = 0; j < nSamples; j++)
            {
                for (std::size_t i = 0; i < nDim; i++)
                {
                    x[l++] = this->prng->gammas[i].dist();
                }
            }
            return true;
        }
        for (int i = 0; i < nSamples; i++)
        {
            x[i] = this->prng->gamma->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gammaDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x, int const nSamples)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            std::size_t const nDim = this->numParams / 2;

#ifdef DEBUG
            if (nDim * nSamples > x.size())
            {
                UMUQFAILRETURN("The input size =", x.size(), " < requested samples size of ", nDim * nSamples, " !");
            }
#endif
            for (std::size_t j = 0, l = 0; j < nSamples; j++)
            {
                for (std::size_t i = 0; i < nDim; i++)
                {
                    x[l++] = this->prng->gammas[i].dist();
                }
            }
            return true;
        }
#ifdef DEBUG
        if (static_cast<std::size_t>(nSamples) > x.size())
        {
            UMUQFAILRETURN("The input size =", x.size(), " < requested samples size of ", nSamples, " !");
        }
#endif
        for (int i = 0; i < nSamples; i++)
        {
            x[i] = this->prng->gamma->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gammaDistribution<DataType, FunctionType>::sample(EMatrixX<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
#ifdef DEBUG
            if (this->numParams / 2 != x.rows())
            {
                UMUQFAILRETURN("The input dimension =", x.rows(), " != samples dimension of ", this->numParams / 2, " !");
            }
#endif
            std::size_t const nDim = this->numParams / 2;

            for (auto j = 0; j < x.cols(); ++j)
            {
                for (std::size_t i = 0; i < nDim; ++i)
                {
                    x(i, j) = this->prng->gammas[i].dist();
                }
            }
            return true;
        }
        for (auto i = 0; i < x.cols(); ++i)
        {
            x(0, i) = this->prng->gamma->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

} // namespace density
} // namespace umuq

#endif // UMUQ_GAMMADISTRIBUTION
