#ifndef UMUQ_EXPONENTIALDISTRIBUTION_H
#define UMUQ_EXPONENTIALDISTRIBUTION_H

namespace umuq
{

inline namespace density
{

/*! \class exponentialDistribution
 * \ingroup Density_Module
 * 
 * \brief The exponential distribution
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
 * It also provides random non-negative values x, distributed according to the exponential 
 * distribution probability density function. 
 * 
 * \note
 * - For using sample member function, setting the the Random Number Generator is required, otherwise, it fails.
 * - Requires that \f$ \mu > 0 \f$. 
 * 
 * \tparam DataType Data type
 */
template <typename DataType, class FunctionType = std::function<DataType(DataType const *)>>
class exponentialDistribution : public densityFunction<DataType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new exponential distribution object
     * 
     * \param mu Mean, \f$ \mu \f$
     */
    explicit exponentialDistribution(DataType const mu);

    /*!
     * \brief Construct a new exponential Distribution object
     * 
     * \param mu Mean, \f$ \mu \f$
     * \param n  Number of input
     */
    explicit exponentialDistribution(DataType const *mu, int const n);

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
    inline DataType exponentialDistribution_f(DataType const *x);

    /*!
     * \brief Log of exponential distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline DataType exponentialDistribution_lf(DataType const *x);

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
     * \brief Create samples of the exponential distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(DataType *x);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<DataType> &x);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EVectorX<DataType> &x);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(DataType *x, int const nSamples);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<DataType> &x, int const nSamples);

    /*!
     * \brief  Create samples of the exponential distribution object
     * 
     * \param x  Matrix of random samples 
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EMatrixX<DataType> &x);

  private:
    /*!
     * \brief Delete an empty exponentialDistribution object construction
     */
    exponentialDistribution() = delete;
};

/*!
 * \brief Construct a new exponential distribution object
 * 
 * \param mu Mean, \f$ \mu \f$
 */
template <typename DataType, class FunctionType>
exponentialDistribution<DataType, FunctionType>::exponentialDistribution(DataType const mu) : densityFunction<DataType, FunctionType>(&mu, 1, "exponential")
{
    this->f = std::bind(&exponentialDistribution<DataType, FunctionType>::exponentialDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&exponentialDistribution<DataType, FunctionType>::exponentialDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
exponentialDistribution<DataType, FunctionType>::exponentialDistribution(DataType const *mu, int const n) : densityFunction<DataType, FunctionType>(mu, n, "exponential")
{
    this->f = std::bind(&exponentialDistribution<DataType, FunctionType>::exponentialDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&exponentialDistribution<DataType, FunctionType>::exponentialDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Exponential distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename DataType, class FunctionType>
inline DataType exponentialDistribution<DataType, FunctionType>::exponentialDistribution_f(DataType const *x)
{
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        if (x[i] < DataType{})
        {
            return DataType{};
        }
    }
    DataType sum(1);
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
template <typename DataType, class FunctionType>
inline DataType exponentialDistribution<DataType, FunctionType>::exponentialDistribution_lf(DataType const *x)
{
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        if (x[i] < DataType{})
        {
            return std::numeric_limits<DataType>::infinity();
        }
    }
    DataType sum(0);
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        sum -= (std::log(this->params[i]) + x[i] / this->params[i]);
    }
    return sum;
}

template <typename DataType, class FunctionType>
inline bool exponentialDistribution<DataType, FunctionType>::setRandomGenerator(psrandom<DataType> *PRNG)
{
    if (PRNG)
    {
        if (PRNG_initialized)
        {
            this->prng = PRNG;
            if (this->numParams > 1)
            {
                return this->prng->set_expns(this->params.data(), this->numParams);
            }
            return this->prng->set_expn(this->params[0]);
        }
        UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to this distribution!");
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename DataType, class FunctionType>
inline psrandom<DataType> *exponentialDistribution<DataType, FunctionType>::getRandomGenerator() { return this->prng; }

template <typename DataType, class FunctionType>
bool exponentialDistribution<DataType, FunctionType>::sample(DataType *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            for (std::size_t i = 0; i < this->numParams; i++)
            {
                x[i] = this->prng->expns[i].dist();
            }
            return true;
        }
        *x = this->prng->expn->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool exponentialDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            for (std::size_t i = 0; i < this->numParams; i++)
            {
                x[i] = this->prng->expns[i].dist();
            }
            return true;
        }
        x[0] = this->prng->expn->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool exponentialDistribution<DataType, FunctionType>::sample(EVectorX<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            for (std::size_t i = 0; i < this->numParams; i++)
            {
                x[i] = this->prng->expns[i].dist();
            }
            return true;
        }
        x[0] = this->prng->expn->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool exponentialDistribution<DataType, FunctionType>::sample(DataType *x, int const nSamples)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            std::size_t const nSizeArray = this->numParams * nSamples;
            for (std::size_t i = 0; i < this->numParams; i++)
            {
                for (std::size_t l = i; l < nSizeArray; l += this->numParams)
                {
                    x[l] = this->prng->expns[i].dist();
                }
            }
            return true;
        }
        for (int i = 0; i < nSamples; i++)
        {
            x[i] = this->prng->expn->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool exponentialDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x, int const nSamples)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            std::size_t const nSizeArray = this->numParams * nSamples;
#ifdef DEBUG
            if (nSizeArray > x.size())
            {
                UMUQFAILRETURN("The input size =", x.size(), " < requested samples size of ", nSizeArray, " !");
            }
#endif
            for (std::size_t i = 0; i < this->numParams; i++)
            {
                for (std::size_t l = i; l < nSizeArray; l += this->numParams)
                {
                    x[l] = this->prng->expns[i].dist();
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
            x[i] = this->prng->expn->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool exponentialDistribution<DataType, FunctionType>::sample(EMatrixX<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            std::size_t const nDim = this->numParams;
#ifdef DEBUG
            if (nDim != x.rows())
            {
                UMUQFAILRETURN("The input dimension =", x.rows(), " != samples dimension of ", nDim, " !");
            }
#endif
            for (auto l = 0; l < x.cols(); ++l)
            {
                for (std::size_t i = 0; i < nDim; i++)
                {
                    x(i, l) = this->prng->expns[i].dist();
                }
            }
            return true;
        }
        for (auto i = 0; i < x.cols(); i++)
        {
            x(0, i) = this->prng->expn->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

} // namespace density
} // namespace umuq

#endif //UMUQ_EXPONENTIALDISTRIBUTION
