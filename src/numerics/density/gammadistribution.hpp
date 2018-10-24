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
 * \tparam T Data type
 */

template <typename T, class V = T const *>
class gammaDistribution : public densityFunction<T, std::function<T(V)>>
{
  public:
    /*!
     * \brief Construct a new Gamma distribution object
     * 
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     */
    gammaDistribution(T const alpha, T const beta = T{1});

    /*!
     * \brief Construct a new gamma Distribution object
     * 
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     * \param n      Total number of alpha + beta inputs
     */
    gammaDistribution(T const *alpha, T const *beta, int const n);

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
    inline T gammaDistribution_f(T const *x);

    /*!
     * \brief Log of Gamma distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T gammaDistribution_lf(T const *x);

    /*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    inline bool setRandomGenerator(psrandom<T> *PRNG);

    /*!
     * \brief Create samples of the Gamma distribution object
     *
     * \param x  Vector of samples
     *
     * \return true
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(T *x);
    bool sample(std::vector<T> &x);
};

template <typename T, class V>
gammaDistribution<T, V>::gammaDistribution(T const alpha, T const beta) : densityFunction<T, std::function<T(V)>>(&alpha, &beta, 2, "gamma")
{
    this->f = std::bind(&gammaDistribution<T, V>::gammaDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gammaDistribution<T, V>::gammaDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
gammaDistribution<T, V>::gammaDistribution(T const *alpha, T const *beta, int const n) : densityFunction<T, std::function<T(V)>>(alpha, beta, n, "gamma")
{
    if (n & 1)
    {
        UMUQFAIL("Wrong number of inputs!")
    }
    this->f = std::bind(&gammaDistribution<T, V>::gammaDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gammaDistribution<T, V>::gammaDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
inline T gammaDistribution<T, V>::gammaDistribution_f(T const *x)
{
    T sum(1);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        if (x[i] < T{})
        {
            return T{};
        }
        else if (x[i] == T{})
        {
            if (this->params[k] == static_cast<T>(1))
            {
                sum *= static_cast<T>(1) / this->params[k + 1];
                continue;
            }
            else
            {
                return T{};
            }
        }
        else if (this->params[k] == static_cast<T>(1))
        {
            sum *= std::exp(-x[i] / this->params[k + 1]) / this->params[k + 1];
        }
        else
        {
            sum *= std::exp((this->params[k] - static_cast<T>(1)) * std::log(x[i] / this->params[k + 1]) - x[i] / this->params[k + 1] - std::lgamma(this->params[k])) / this->params[k + 1];
        }
    }
    return sum;
}

template <typename T, class V>
inline T gammaDistribution<T, V>::gammaDistribution_lf(T const *x)
{
    for (std::size_t i = 0; i < this->numParams / 2; i++)
    {
        if (x[i] < T{})
        {
            return -std::numeric_limits<T>::infinity();
        }
    }
    T sum(0);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        sum += -std::lgamma(this->params[k]) - this->params[k] * std::log(this->params[k + 1]) + (this->params[k] - static_cast<T>(1)) * std::log(x[i]) - x[i] / this->params[k + 1];
    }
    return sum;
}

template <typename T, class V>
inline bool gammaDistribution<T, V>::setRandomGenerator(psrandom<T> *PRNG)
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

template <typename T, class V>
bool gammaDistribution<T, V>::sample(T *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (int i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->gammas[i].dist();
            }
            return true;
        }
        *x = this->prng->gamma->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!")
#endif
}

template <typename T, class V>
bool gammaDistribution<T, V>::sample(std::vector<T> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (int i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->gammas[i].dist();
            }
            return true;
        }
        x[0] = this->prng->gamma->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!")
#endif
}

} // namespace density
} // namespace umuq

#endif // UMUQ_GAMMADISTRIBUTION
