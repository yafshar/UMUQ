#ifndef UMUQ_GAUSSIANDISTRIBUTION_H
#define UMUQ_GAUSSIANDISTRIBUTION_H

namespace umuq
{

inline namespace density
{

/*! \class gaussianDistribution
 * \ingroup Density_Module
 * 
 * \brief The Gaussian distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a Gaussian 
 * distribution with standard deviation \f$ \sigma \f$ <br>
 * using:
 * 
 * \f$
 * p(x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(-\frac{\left(x - \mu \right)^2}{2\sigma^2}\right)}.
 * \f$
 * 
 * It also provides random values x, distributed according to the Gaussian distribution probability 
 * density function. 
 * 
 * \note
 * - For using sample member function, setting the the Random Number Generator is required, otherwise, it fails.
 * 
 * \tparam DataType Data type
 */
template <typename DataType, class FunctionType = std::function<DataType(DataType const *)>>
class gaussianDistribution : public densityFunction<DataType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new gaussian Distribution object
     * 
     * \param mu     Mean, \f$ \mu \f$
     * \param sigma  Standard deviation \f$ \sigma \f$
     */
    gaussianDistribution(DataType const mu, DataType const sigma);

    /*!
     * \brief Construct a new gaussian Distribution object
     * 
     * \param mu     Mean, \f$ \mu \f$
     * \param sigma  Standard deviation \f$ \sigma \f$
     * \param n      Total number of Mean + Standard deviation inputs
     */
    gaussianDistribution(DataType const *mu, DataType const *sigma, int const n);

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
    inline DataType gaussianDistribution_f(DataType const *x);

    /*!
     * \brief Log of Gaussian Distribution density function
     * 
     * \param x  Input value
     * 
     * \returns  Log of density function value 
     */
    inline DataType gaussianDistribution_lf(DataType const *x);

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
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(DataType *x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<DataType> &x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Vector of samples
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EVectorX<DataType> &x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(DataType *x, int const nSamples);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<DataType> &x, int const nSamples);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Matrix of random samples 
     *
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EMatrixX<DataType> &x);
};

template <typename DataType, class FunctionType>
gaussianDistribution<DataType, FunctionType>::gaussianDistribution(DataType const mu, DataType const sigma) : densityFunction<DataType, FunctionType>(&mu, &sigma, 2, "gaussian")
{
    this->f = std::bind(&gaussianDistribution<DataType, FunctionType>::gaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gaussianDistribution<DataType, FunctionType>::gaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
gaussianDistribution<DataType, FunctionType>::gaussianDistribution(DataType const *mu, DataType const *sigma, int const n) : densityFunction<DataType, FunctionType>(mu, sigma, n, "gaussian")
{
    if (n & 1)
    {
        UMUQFAIL("Wrong number of inputs!");
    }
    this->f = std::bind(&gaussianDistribution<DataType, FunctionType>::gaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&gaussianDistribution<DataType, FunctionType>::gaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
inline DataType gaussianDistribution<DataType, FunctionType>::gaussianDistribution_f(DataType const *x)
{
    DataType sum(1);
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        DataType const xSigma = (x[i] - this->params[k]) / this->params[k + 1];
        sum *= static_cast<DataType>(1) / (M_S2PI * this->params[k + 1]) * std::exp(-0.5 * xSigma * xSigma);
    }
    return sum;
}

template <typename DataType, class FunctionType>
inline DataType gaussianDistribution<DataType, FunctionType>::gaussianDistribution_lf(DataType const *x)
{
    DataType sum(0);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        DataType const xSigma = (x[i] - this->params[k]) / this->params[k + 1];
        sum += -0.5 * M_L2PI - std::log(this->params[k + 1]) - 0.5 * xSigma * xSigma;
    }
    return sum;
}

template <typename DataType, class FunctionType>
inline bool gaussianDistribution<DataType, FunctionType>::setRandomGenerator(psrandom<DataType> *PRNG)
{
    if (PRNG)
    {
        if (PRNG_initialized)
        {
            this->prng = PRNG;
            if (this->numParams > 2)
            {
                return this->prng->set_normals(this->params.data(), this->numParams);
            }
            return this->prng->set_normal(this->params[0], this->params[1]);
        }
        UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to this distribution!");
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename DataType, class FunctionType>
inline psrandom<DataType> *gaussianDistribution<DataType, FunctionType>::getRandomGenerator() { return this->prng; }

template <typename DataType, class FunctionType>
bool gaussianDistribution<DataType, FunctionType>::sample(DataType *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (std::size_t i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->normals[i].dist();
            }
            return true;
        }
        *x = this->prng->normal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gaussianDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (std::size_t i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->normals[i].dist();
            }
            return true;
        }
        x[0] = this->prng->normal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gaussianDistribution<DataType, FunctionType>::sample(EVectorX<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            for (std::size_t i = 0; i < this->numParams / 2; i++)
            {
                x[i] = this->prng->normals[i].dist();
            }
            return true;
        }
        x[0] = this->prng->normal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gaussianDistribution<DataType, FunctionType>::sample(DataType *x, int const nSamples)
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
                    x[l++] = this->prng->normals[i].dist();
                }
            }
            return true;
        }
        for (int i = 0; i < nSamples; i++)
        {
            x[i] = this->prng->normal->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gaussianDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x, int const nSamples)
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
                    x[l] = this->prng->normals[i].dist();
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
            x[i] = this->prng->normal->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool gaussianDistribution<DataType, FunctionType>::sample(EMatrixX<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        if (this->numParams > 2)
        {
            std::size_t const nDim = this->numParams / 2;

            for (auto j = 0; j < x.cols(); j++)
            {
                for (std::size_t i = 0; i < nDim; i++)
                {
                    x(i, j) = this->prng->normals[i].dist();
                }
            }
            return true;
        }
        for (auto i = 0; i < x.cols(); i++)
        {
            x(0, i) = this->prng->normal->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

} // namespace density
} // namespace umuq

#endif //UMUQ_GAUSSIANDISTRIBUTION
