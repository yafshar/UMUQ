#ifndef UMUQ_UNIFORMDISTRIBUTION_H
#define UMUQ_UNIFORMDISTRIBUTION_H

namespace umuq
{

inline namespace density
{

/*! \class uniformDistribution
 * \ingroup Density_Module
 * 
 * \brief Flat (Uniform) distribution function
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a uniform distribution 
 * from \f$ [a \cdots b] \f$, <br>
 * using:
 * 
 * \f$
 * p(x)= \left\{
 * \begin{matrix}
 * 1/(b-a)  &a \leq x < b \\ 
 *  0       &otherwise
 * \end{matrix}
 * \right.
 * \f$
 * 
 * \note
 * - For using sample member function, setting the the Random Number Generator is required, otherwise, it fails.
 * 
 * \tparam T Data type
 */
template <typename T, class V = T const *>
class uniformDistribution : public densityFunction<T, std::function<T(V)>>
{
  public:
    /*!
     * \brief Construct a new uniform Distribution object
     * 
     * \param a  Lower bound
     * \param b  Upper bound
     */
    uniformDistribution(T const a, T const b);

    /*!
     * \brief Construct a new uniform Distribution object
     * 
     * \param a  Lower bound
     * \param b  Upper bound
     * \param n  Total number of Lower bound + Upper bound inputs
     */
    uniformDistribution(T const *a, T const *b, int const n);

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
    inline T uniformDistribution_f(T const *x);

    /*!
     * \brief Log of Uniform distribution density function
     * 
     * \param x  Input value
     * 
     * \returns  Log of density function value
     */
    inline T uniformDistribution_lf(T const *x);

    /*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
     * 
     * \return false If it encounters an unexpected problem
     */
    inline bool setRandomGenerator(psrandom<T> *PRNG);

    /*!
     * \brief Get the Random Number Generator object 
     * 
     * \returns Pseudo-random number object. \sa umuq::random::psrandom.
     */
    inline psrandom<T> *getRandomGenerator();

    /*!
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x  Vector of samples
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(T *x);

    /*!
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x  Vector of samples
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<T> &x);

    /*!
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x  Vector of samples
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EVectorX<T> &x);

    /*!
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(T *x, int const nSamples);

    /*!
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<T> &x, int const nSamples);

    /*!
     * \brief  Create samples of the uniform Distribution object
     * 
     * \param x  Matrix of random samples 
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EMatrixX<T> &x);

  private:
    //! Const value for uniform distribution function
    T uniformDistribution_fValue;
    //! Const value for logarithm of the uniform distribution function
    T uniformDistribution_lfValue;

    //! Helper function
    inline T uniformDistribution_f_();
    //! Helper log function
    inline T uniformDistribution_lf_();
};

template <typename T, class V>
uniformDistribution<T, V>::uniformDistribution(T const a, T const b) : densityFunction<T, std::function<T(V)>>(&a, &b, 2, "uniform")
{
    this->f = std::bind(&uniformDistribution<T>::uniformDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<T>::uniformDistribution_lf, this, std::placeholders::_1);
    uniformDistribution_fValue = uniformDistribution_f_();
    uniformDistribution_lfValue = uniformDistribution_lf_();
}

template <typename T, class V>
uniformDistribution<T, V>::uniformDistribution(T const *a, T const *b, int const n) : densityFunction<T, std::function<T(V)>>(a, b, n, "uniform")
{
    if (n & 1)
    {
        UMUQFAIL("Wrong number of inputs!");
    }
    this->f = std::bind(&uniformDistribution<T>::uniformDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<T>::uniformDistribution_lf, this, std::placeholders::_1);
    uniformDistribution_fValue = uniformDistribution_f_();
    uniformDistribution_lfValue = uniformDistribution_lf_();
}

template <typename T, class V>
uniformDistribution<T, V>::~uniformDistribution() {}

template <typename T, class V>
inline T uniformDistribution<T, V>::uniformDistribution_f(T const *x)
{
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        if (x[i] < this->params[k] || x[i] >= this->params[k + 1])
        {
            return T{};
        }
    }
    return uniformDistribution_fValue;
}

template <typename T, class V>
inline T uniformDistribution<T, V>::uniformDistribution_lf(T const *x)
{
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        if (x[i] < this->params[k] || x[i] >= this->params[k + 1])
        {
            return std::numeric_limits<T>::infinity();
        }
    }
    return uniformDistribution_lfValue;
}

template <typename T, class V>
inline T uniformDistribution<T, V>::uniformDistribution_f_()
{
    T sum(1);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        sum *= static_cast<T>(1) / (this->params[k + 1] - this->params[k]);
    }
    return sum;
}

template <typename T, class V>
inline T uniformDistribution<T, V>::uniformDistribution_lf_()
{
    T sum(0);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        sum -= std::log(this->params[k + 1] - this->params[k]);
    }
    return sum;
}

template <typename T, class V>
inline bool uniformDistribution<T, V>::setRandomGenerator(psrandom<T> *PRNG)
{
    if (PRNG)
    {
        if (PRNG_initialized)
        {
            this->prng = PRNG;
            if (this->numParams > 2)
            {
                return this->prng->set_uniforms(this->params.data(), this->numParams);
            }
            return this->prng->set_uniform(this->params[0], this->params[1]);
        }
        UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to this distribution!");
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename T, class V>
inline psrandom<T> *uniformDistribution<T, V>::getRandomGenerator() { return this->prng; }

template <typename T, class V>
bool uniformDistribution<T, V>::sample(T *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (std::size_t i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->uniforms[i].dist();
            }
            return true;
        }
        *x = this->prng->uniform->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename T, class V>
bool uniformDistribution<T, V>::sample(std::vector<T> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (std::size_t i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->uniforms[i].dist();
            }
            return true;
        }
        x[0] = this->prng->uniform->dist();
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename T, class V>
bool uniformDistribution<T, V>::sample(EVectorX<T> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (std::size_t i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->uniforms[i].dist();
            }
            return true;
        }
        x[0] = this->prng->uniform->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename T, class V>
bool uniformDistribution<T, V>::sample(T *x, int const nSamples)
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
                    x[l++] = this->prng->uniforms[i].dist();
                }
            }
            return true;
        }
        for (int i = 0; i < nSamples; i++)
        {
            x[i] = this->prng->uniform->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename T, class V>
bool uniformDistribution<T, V>::sample(std::vector<T> &x, int const nSamples)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            std::size_t const Stride = this->numParams / 2;
            std::size_t const nSizeArray = Stride * nSamples;

#ifdef DEBUG
            if (nSizeArray > x.size())
            {
                UMUQFAILRETURN("The input size =", x.size(), " < requested samples size of ", nSizeArray, " !");
            }
#endif
            for (std::size_t i = 0; i < Stride; i++)
            {
                for (std::size_t l = i; l < nSizeArray; l += Stride)
                {
                    x[l] = this->prng->uniforms[i].dist();
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
            x[i] = this->prng->uniform->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename T, class V>
bool uniformDistribution<T, V>::sample(EMatrixX<T> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
        if (this->numParams / 2 != x.rows())
        {
            UMUQFAILRETURN("The input dimension =", x.rows(), " != samples dimension of ", this->numParams / 2, " !");
        }
#endif
        if (this->numParams > 2)
        {
            std::size_t const nDim = this->numParams / 2;

            for (auto j = 0; j < x.cols(); j++)
            {
                for (std::size_t i = 0; i < nDim; i++)
                {
                    x(i, j) = this->prng->uniforms[i].dist();
                }
            }
            return true;
        }
        for (auto i = 0; i < x.cols(); i++)
        {
            x(0, i) = this->prng->uniform->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

} // namespace density
} // namespace umuq

#endif //UMUQ_UNIFORMDISTRIBUTION_H