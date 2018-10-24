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
 * \tparam T Data type
 */
template <typename T, class V = T const *>
class exponentialDistribution : public densityFunction<T, std::function<T(V)>>
{
  public:
    /*!
     * \brief Construct a new exponential distribution object
     * 
     * \param mu Mean, \f$ \mu \f$
     */
    explicit exponentialDistribution(T const mu);

    /*!
     * \brief Construct a new exponential Distribution object
     * 
     * \param mu Mean, \f$ \mu \f$
     * \param n  Number of input
     */
    explicit exponentialDistribution(T const *mu, int const n);

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
    inline T exponentialDistribution_f(T const *x);

    /*!
     * \brief Log of exponential distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T exponentialDistribution_lf(T const *x);

    /*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
     * 
     * \return false If it encounters an unexpected problem
     */
    inline bool setRandomGenerator(psrandom<T> *PRNG);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(T *x);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<T> &x);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(T *x, int const nSamples);

    /*!
     * \brief Create samples of the exponential distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<T> &x, int const nSamples);

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
template <typename T, class V>
exponentialDistribution<T, V>::exponentialDistribution(T const mu) : densityFunction<T, std::function<T(V)>>(&mu, 1, "exponential")
{
    this->f = std::bind(&exponentialDistribution<T, V>::exponentialDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&exponentialDistribution<T, V>::exponentialDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
exponentialDistribution<T, V>::exponentialDistribution(T const *mu, int const n) : densityFunction<T, std::function<T(V)>>(mu, n, "exponential")
{
    this->f = std::bind(&exponentialDistribution<T, V>::exponentialDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&exponentialDistribution<T, V>::exponentialDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Exponential distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T, class V>
inline T exponentialDistribution<T, V>::exponentialDistribution_f(T const *x)
{
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        if (x[i] < T{})
        {
            return T{};
        }
    }
    T sum(1);
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
template <typename T, class V>
inline T exponentialDistribution<T, V>::exponentialDistribution_lf(T const *x)
{
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        if (x[i] < T{})
        {
            return std::numeric_limits<T>::infinity();
        }
    }
    T sum(0);
    for (std::size_t i = 0; i < this->numParams; i++)
    {
        sum -= (std::log(this->params[i]) + x[i] / this->params[i]);
    }
    return sum;
}

template <typename T, class V>
inline bool exponentialDistribution<T, V>::setRandomGenerator(psrandom<T> *PRNG)
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

template <typename T, class V>
bool exponentialDistribution<T, V>::sample(T *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            for (int i = 0; i < this->numParams; i++)
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

template <typename T, class V>
bool exponentialDistribution<T, V>::sample(std::vector<T> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            for (int i = 0; i < this->numParams; i++)
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

template <typename T, class V>
bool exponentialDistribution<T, V>::sample(T *x, int const nSamples)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            int const nSizeArray = this->numParams * nSamples;
            for (int i = 0; i < this->numParams; i++)
            {
                for (int l = i; l < nSizeArray; l += this->numParams)
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

template <typename T, class V>
bool exponentialDistribution<T, V>::sample(std::vector<T> &x, int const nSamples)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 1)
        {
            int const nSizeArray = this->numParams * nSamples;
#ifdef DEBUG
            if (static_cast<std::size_t>(nSizeArray) > x.size())
            {
                UMUQFAILRETURN("The input size =", x.size(), " < requested samples size of ", nSizeArray, " !");
            }
#endif
            for (int i = 0; i < this->numParams; i++)
            {
                for (int l = i; l < nSizeArray; l += this->numParams)
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

} // namespace density
} // namespace umuq

#endif //UMUQ_EXPONENTIALDISTRIBUTION
