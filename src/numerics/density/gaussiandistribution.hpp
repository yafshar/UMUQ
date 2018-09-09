#ifndef UMUQ_GAUSSIANDISTRIBUTION_H
#define UMUQ_GAUSSIANDISTRIBUTION_H

namespace umuq
{
/*! \namespace density
 * \brief Namespace containing all the functions for probability density computation
 *
 */
inline namespace density
{

/*! \class gaussianDistribution
 * \brief The Gaussian distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a Gaussian 
 * distribution wiuh standard deviation \f$ \sigma \f$
 * using: 
 * \f[
 * p(x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(-\frac{\left(x - \mu \right)^2}{2\sigma^2}\right)}.
 * \f]
 * 
 * It also provides random values x, distributed according to the Gaussian distribution probability 
 * density function. 
 * 
 * 
 * \tparam T Data type
 */
template <typename T, class V = T const *>
class gaussianDistribution : public densityFunction<T, std::function<T(V)>>
{
  public:
    /*!
     * \brief Construct a new gaussian Distribution object
     * 
     * \param mu     Mean, \f$ \mu \f$
     * \param sigma  Standard deviation \f$ \sigma \f$
     */
    gaussianDistribution(T const mu, T const sigma);

    /*!
     * \brief Construct a new gaussian Distribution object
     * 
     * \param mu     Mean, \f$ \mu \f$
     * \param sigma  Standard deviation \f$ \sigma \f$
     * \param n      Total number of Mean + Standard deviation inputs
     */
    gaussianDistribution(T const *mu, T const *sigma, int const n);

    /*!
     * \brief Destroy the gaussian Distribution object
     * 
     */
    ~gaussianDistribution() {}

    /*!
     * \brief Gaussian Distribution density function
     * 
     * \param x  Input value
     * 
     * \returns Density function value 
     */
    inline T gaussianDistribution_f(T const *x);

    /*!
     * \brief Log of Gaussian Distribution density function
     * 
     * \param x  Input value
     * 
     * \returns  Log of density function value 
     */
    inline T gaussianDistribution_lf(T const *x);

    /*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object \sa psrandom
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    inline bool setRandomGenerator(psrandom<T> *PRNG);

    /*!
     * \brief Create samples of the Gaussian Distribution object
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
gaussianDistribution<T, V>::gaussianDistribution(T const mu, T const sigma) : densityFunction<T, std::function<T(V)>>(&mu, &sigma, 2, "gaussian")
{
    this->f = std::bind(&gaussianDistribution<T, V>::gaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gaussianDistribution<T, V>::gaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
gaussianDistribution<T, V>::gaussianDistribution(T const *mu, T const *sigma, int const n) : densityFunction<T, std::function<T(V)>>(mu, sigma, n, "gaussian")
{
    if (n % 2 != 0)
    {
        UMUQFAIL("Wrong number of inputs!");
    }
    this->f = std::bind(&gaussianDistribution<T, V>::gaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gaussianDistribution<T, V>::gaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
inline T gaussianDistribution<T, V>::gaussianDistribution_f(T const *x)
{
    T sum(1);
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        T const xSigma = (x[i] - this->params[k]) / this->params[k + 1];
        sum *= static_cast<T>(1) / (M_S2PI * this->params[k + 1]) * std::exp(-0.5 * xSigma * xSigma);
    }
    return sum;
}

template <typename T, class V>
inline T gaussianDistribution<T, V>::gaussianDistribution_lf(T const *x)
{
    T sum(0);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        T const xSigma = (x[i] - this->params[k]) / this->params[k + 1];
        sum += -0.5 * M_L2PI - std::log(this->params[k + 1]) - 0.5 * xSigma * xSigma;
    }
    return sum;
}

template <typename T, class V>
inline bool gaussianDistribution<T, V>::setRandomGenerator(psrandom<T> *PRNG)
{
    if (PRNG)
    {
        this->prng = PRNG;
        if (this->numParams > 2)
        {
            return this->prng->set_normals(this->params.data(), this->numParams);
        }
        return this->prng->set_normal(this->params[0], this->params[1]);
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename T, class V>
bool gaussianDistribution<T, V>::sample(T *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (int i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->normals[i].dist();
            }
            return true;
        }
        *x = this->prng->normal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!")
#endif
}

template <typename T, class V>
bool gaussianDistribution<T, V>::sample(std::vector<T> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (int i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->normals[i].dist();
            }
            return true;
        }
        x[0] = this->prng->normal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!")
#endif
}

} // namespace density
} // namespace umuq

#endif //UMUQ_GAUSSIANDISTRIBUTION_H