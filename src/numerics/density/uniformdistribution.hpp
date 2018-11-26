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
 * \tparam DataType Data type
 */
template <typename DataType, class FunctionType = std::function<DataType(DataType const *)>>
class uniformDistribution : public densityFunction<DataType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new uniform Distribution object
     * 
     * \param a  Lower bound
     * \param b  Upper bound
     */
    uniformDistribution(DataType const a, DataType const b);

    /*!
     * \brief Construct a new uniform Distribution object
     * 
     * \param a  Lower bound
     * \param b  Upper bound
     * \param n  Total number of Lower bound + Upper bound inputs
     */
    uniformDistribution(DataType const *a, DataType const *b, int const n);

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
    inline DataType uniformDistribution_f(DataType const *x);

    /*!
     * \brief Log of Uniform distribution density function
     * 
     * \param x  Input value
     * 
     * \returns  Log of density function value
     */
    inline DataType uniformDistribution_lf(DataType const *x);

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
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x  Vector of samples
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(DataType *x);

    /*!
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x  Vector of samples
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<DataType> &x);

    /*!
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x  Vector of samples
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EVectorX<DataType> &x);

    /*!
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(DataType *x, int const nSamples);

    /*!
     * \brief Create samples of the uniform Distribution object
     * 
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(std::vector<DataType> &x, int const nSamples);

    /*!
     * \brief  Create samples of the uniform Distribution object
     * 
     * \param x  Matrix of random samples 
     * 
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(EMatrixX<DataType> &x);

  private:
    //! Const value for uniform distribution function
    DataType uniformDistribution_fValue;
    //! Const value for logarithm of the uniform distribution function
    DataType uniformDistribution_lfValue;

    //! Helper function
    inline DataType uniformDistribution_f_();
    //! Helper log function
    inline DataType uniformDistribution_lf_();
};

template <typename DataType, class FunctionType>
uniformDistribution<DataType, FunctionType>::uniformDistribution(DataType const a, DataType const b) : densityFunction<DataType, FunctionType>(&a, &b, 2, "uniform")
{
    this->f = std::bind(&uniformDistribution<DataType>::uniformDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<DataType>::uniformDistribution_lf, this, std::placeholders::_1);
    uniformDistribution_fValue = uniformDistribution_f_();
    uniformDistribution_lfValue = uniformDistribution_lf_();
}

template <typename DataType, class FunctionType>
uniformDistribution<DataType, FunctionType>::uniformDistribution(DataType const *a, DataType const *b, int const n) : densityFunction<DataType, FunctionType>(a, b, n, "uniform")
{
    if (n & 1)
    {
        UMUQFAIL("Wrong number of inputs!");
    }
    this->f = std::bind(&uniformDistribution<DataType>::uniformDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<DataType>::uniformDistribution_lf, this, std::placeholders::_1);
    uniformDistribution_fValue = uniformDistribution_f_();
    uniformDistribution_lfValue = uniformDistribution_lf_();
}

template <typename DataType, class FunctionType>
uniformDistribution<DataType, FunctionType>::~uniformDistribution() {}

template <typename DataType, class FunctionType>
inline DataType uniformDistribution<DataType, FunctionType>::uniformDistribution_f(DataType const *x)
{
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        if (x[i] < this->params[k] || x[i] >= this->params[k + 1])
        {
            return DataType{};
        }
    }
    return uniformDistribution_fValue;
}

template <typename DataType, class FunctionType>
inline DataType uniformDistribution<DataType, FunctionType>::uniformDistribution_lf(DataType const *x)
{
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        if (x[i] < this->params[k] || x[i] >= this->params[k + 1])
        {
            return std::numeric_limits<DataType>::infinity();
        }
    }
    return uniformDistribution_lfValue;
}

template <typename DataType, class FunctionType>
inline DataType uniformDistribution<DataType, FunctionType>::uniformDistribution_f_()
{
    DataType sum(1);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        sum *= static_cast<DataType>(1) / (this->params[k + 1] - this->params[k]);
    }
    return sum;
}

template <typename DataType, class FunctionType>
inline DataType uniformDistribution<DataType, FunctionType>::uniformDistribution_lf_()
{
    DataType sum(0);
    for (std::size_t i = 0, k = 0; k < this->numParams; i++, k += 2)
    {
        sum -= std::log(this->params[k + 1] - this->params[k]);
    }
    return sum;
}

template <typename DataType, class FunctionType>
inline bool uniformDistribution<DataType, FunctionType>::setRandomGenerator(psrandom<DataType> *PRNG)
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

template <typename DataType, class FunctionType>
inline psrandom<DataType> *uniformDistribution<DataType, FunctionType>::getRandomGenerator() { return this->prng; }

template <typename DataType, class FunctionType>
bool uniformDistribution<DataType, FunctionType>::sample(DataType *x)
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

template <typename DataType, class FunctionType>
bool uniformDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x)
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

template <typename DataType, class FunctionType>
bool uniformDistribution<DataType, FunctionType>::sample(EVectorX<DataType> &x)
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

template <typename DataType, class FunctionType>
bool uniformDistribution<DataType, FunctionType>::sample(DataType *x, int const nSamples)
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

template <typename DataType, class FunctionType>
bool uniformDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x, int const nSamples)
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

template <typename DataType, class FunctionType>
bool uniformDistribution<DataType, FunctionType>::sample(EMatrixX<DataType> &x)
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