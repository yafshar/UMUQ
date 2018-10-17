#ifndef UMUQ_PSRANDOM_EXPONENTIALDISTRIBUTION_H
#define UMUQ_PSRANDOM_EXPONENTIALDISTRIBUTION_H

namespace umuq
{

namespace randomdist
{

/*! \class exponentialDistribution
 * \ingroup Random_Module
 * 
 * \brief Generates random non-negative values x, distributed according to probability density function \f$ \frac{1}{\mu}e^{\left(-\frac{x}{\mu}\right)} \f$ 
 * 
 * This class provides random non-negative values x, distributed according to the probability 
 * density function of: <br>
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
 * where \f$ \lambda > 0 \f$ is rate. Thus we have \f$ \lambda = \frac{1}{\mu} \f$
 * 
 * \note
 * - This should be called after setting the State of psrandom object
 * - Requires that \f$ \mu > 0 \f$. 
 * 
 */
template <typename T>
class exponentialDistribution
{
  public:
    /*!
     * \brief Construct a new exponential Distribution object
     * 
     */
    exponentialDistribution();

    /*!
     * \brief Construct a new exponential distribution object
     * 
     * \param mu Mean, \f$ \mu \f$
     */
    explicit exponentialDistribution(T const mu);

    /*!
     * \brief Move constructor, construct a new exponentialDistribution object from input exponentialDistribution object
     * 
     * \param other  Input exponentialDistribution object
     */
    exponentialDistribution(exponentialDistribution<T> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other  Input exponentialDistribution object
     * \return exponentialDistribution& 
     */
    exponentialDistribution<T> &operator=(exponentialDistribution<T> &&other);

    /*!
     * \brief Random numbers x according to probability density function \f$ \frac{1}{\mu}e^{\left(-\frac{x}{\mu}\right)} \f$ <br>
     * The result type generated by the generator is undefined if T is not one of float, 
     * double, or long double
     * 
     * \return Random numbers x according to probability density function \f$ \frac{1}{\mu}e^{\left(-\frac{x}{\mu}\right)} \f$ 
     * 
     */
    inline T operator()();

    /*!
     * \brief Random numbers x according to probability density function \f$ \frac{1}{\mu}e^{\left(-\frac{x}{\mu}\right)} \f$ <br>
     * The result type generated by the generator is undefined if T is not one of float, 
     * double, or long double
     * 
     * \return Random numbers x according to probability density function \f$ \frac{1}{\mu}e^{\left(-\frac{x}{\mu}\right)} \f$ 
     * 
     */
    inline T dist();

  private:
    /*!
     * \brief Delete a exponentialDistribution object copy construction
     * 
     * Make it noncopyable.
     */
    exponentialDistribution(exponentialDistribution<T> const &) = delete;

    /*!
     * \brief Delete a exponentialDistribution object assignment
     * 
     * Make it nonassignable
     * 
     * \returns exponentialDistribution<T>& 
     */
    exponentialDistribution<T> &operator=(exponentialDistribution<T> const &) = delete;

  private:
    /*!
     * \brief STL implementation of the random numbers according to to probability density function \f$ \lambda e^{-\lambda x} \f$, <br>
     * here \f$ \lambda > 0 \f$ is a rate and we have \f$ \lambda = \frac{1}{\mu} \f$
     */
    std::exponential_distribution<T> d;
};

template <typename T>
exponentialDistribution<T>::exponentialDistribution() : d(T{1})
{
    if (!PRNG_initialized)
    {
        UMUQFAIL("One should set the current state of the engine before constructing this object!");
    }
}

template <typename T>
exponentialDistribution<T>::exponentialDistribution(T const mu) : d(1. / mu)
{
    if (!PRNG_initialized)
    {
        UMUQFAIL("One should set the current state of the engine before constructing this object!");
    }
    if (mu <= 0)
    {
        UMUQFAIL("It requires that ", mu, "> 0.!");
    }
}

template <typename T>
exponentialDistribution<T>::exponentialDistribution(exponentialDistribution<T> &&other) : d(std::move(other.d)) {}

template <typename T>
exponentialDistribution<T> &exponentialDistribution<T>::operator=(exponentialDistribution<T> &&other)
{
    d = std::move(other.d);
    return *this;
}

template <typename T>
inline T exponentialDistribution<T>::operator()()
{
    // Get the thread ID
    int const me = torc_i_worker_id();
    return d(NumberGenerator[me]);
}

template <typename T>
inline T exponentialDistribution<T>::dist()
{
    // Get the thread ID
    int const me = torc_i_worker_id();
    return d(NumberGenerator[me]);
}

} // namespace randomdist
} // namespace umuq

#endif // UMUQ_PSRANDOM_EXPONENTIALDISTRIBUTION
