#ifndef UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION_H
#define UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION_H

namespace umuq
{

inline namespace density
{

/*! \class multivariateGaussianDistribution
 * \ingroup Density_Module
 * 
 * \brief The Multivariate Gaussian Distribution
 * 
 * The Multivariate Gaussian Distribution is a generalization of the one-dimensional (univariate) Gaussian 
 * distribution to higher dimensions. One definition is that a random vector is said to be k-variate normally 
 * distributed if every linear combination of its k components has a univariate Gaussian distribution.
 * 
 * Reference:<br>
 * https://en.wikipedia.org/wiki/Multivariate_normal_distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x using mean vector imean
 * and variance-covariance matrix icovariance. <br>
 * using:
 * 
 * \f$
 * p(x_1,\cdots,x_k) = \frac{1}{\sqrt{(2 \pi)^k |\Sigma|}} \exp \left(-\frac{1}{2} (x - \mu)^DataType \Sigma^{-1} (x - \mu)\right).
 * \f$
 * 
 * It also provides random values x, distributed according to the Multivariate Gaussian Distribution probability 
 * density function. 
 * 
 * \note
 * - For using any member function, a pointer to a Random Number Generator object 
 *   in the construction is required, otherwise, it fails.<br>
 *   \sa umuq::random::psrandom. 
 * 
 * \tparam DataType Data type
 */
template <typename DataType, class FunctionType = std::function<DataType(DataType const *)>>
class multivariateGaussianDistribution : public densityFunction<DataType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new multivariategaussian distribution object
     *
     * \param imean        Mean vector of size \f$n\f$
     * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
     * \param PRNG         Pseudo-random number object. \sa umuq::random::psrandom.
     */
    multivariateGaussianDistribution(EVectorX<DataType> const &imean, EMatrixX<DataType> const &icovariance, psrandom<DataType> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object
     * 
     * \param imean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     * \param PRNG         Pseudo-random number object. \sa umuq::random::psrandom.
     */
    multivariateGaussianDistribution(DataType const *imean, DataType const *icovariance, int const n, psrandom<DataType> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param PRNG         Pseudo-random number object. \sa umuq::random::psrandom.
     */
    explicit multivariateGaussianDistribution(EMatrixX<DataType> const &icovariance, psrandom<DataType> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     * 
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     * \param PRNG         Pseudo-random number object. \sa umuq::random::psrandom.
     */
    multivariateGaussianDistribution(DataType const *icovariance, int const n, psrandom<DataType> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0, covariance=I)
     * 
     * \param n vector size
     * \param PRNG     Pseudo-random number object. \sa umuq::random::psrandom.
     */
    explicit multivariateGaussianDistribution(int const n, psrandom<DataType> *PRNG = nullptr);

    /*!
     * \brief Destroy the multinomial distribution object
     * 
     */
    ~multivariateGaussianDistribution();

    /*!
     * \brief Multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param x  Input vector of size \f$ n \f$
     * 
     * \returns Density function value 
     */
    inline DataType multivariategaussianDistribution_f(DataType const *x);

    /*!
     * \brief Log of multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param x  Input vector of size \f$ n \f$
     * 
     * \returns  Log of density function value 
     */
    inline DataType multivariategaussianDistribution_lf(DataType const *x);

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

  private:
    /*!
     * \brief Multivariate random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::multivariateNormalDistribution<DataType>> mvnormal;
};

template <typename DataType, class FunctionType>
multivariateGaussianDistribution<DataType, FunctionType>::multivariateGaussianDistribution(EVectorX<DataType> const &imean, EMatrixX<DataType> const &icovariance, psrandom<DataType> *PRNG) : mvnormal(nullptr)
{
    if (PRNG)
    {
        if (!PRNG_initialized)
        {
            UMUQFAIL("One should set the state of the pseudo random number generator before setting it to this distribution!");
        }
        this->prng = PRNG;
        if (!this->prng->set_mvnormal(imean, icovariance))
        {
            UMUQFAIL("Failed to set mvnormal object!");
        }
    }
    else
    {
        try
        {
            mvnormal.reset(new randomdist::multivariateNormalDistribution<DataType>(imean, icovariance));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = imean.size();
    this->f = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
multivariateGaussianDistribution<DataType, FunctionType>::multivariateGaussianDistribution(DataType const *imean, DataType const *icovariance, int const n, psrandom<DataType> *PRNG) : mvnormal(nullptr)
{
    if (PRNG)
    {
        if (!PRNG_initialized)
        {
            UMUQFAIL("One should set the state of the pseudo random number generator before setting it to this distribution!");
        }
        this->prng = PRNG;
        if (!this->prng->set_mvnormal(imean, icovariance, n))
        {
            UMUQFAIL("Failed to set mvnormal object!");
        }
    }
    else
    {
        try
        {
            mvnormal.reset(new randomdist::multivariateNormalDistribution<DataType>(imean, icovariance, n));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
multivariateGaussianDistribution<DataType, FunctionType>::multivariateGaussianDistribution(EMatrixX<DataType> const &icovariance, psrandom<DataType> *PRNG) : mvnormal(nullptr)
{
    if (PRNG)
    {
        if (!PRNG_initialized)
        {
            UMUQFAIL("One should set the state of the pseudo random number generator before setting it to this distribution!");
        }
        this->prng = PRNG;
        if (!this->prng->set_mvnormal(icovariance))
        {
            UMUQFAIL("Failed to set mvnormal object!");
        }
    }
    else
    {
        try
        {
            mvnormal.reset(new randomdist::multivariateNormalDistribution<DataType>(icovariance));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = icovariance.rows();
    this->f = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
multivariateGaussianDistribution<DataType, FunctionType>::multivariateGaussianDistribution(DataType const *icovariance, int const n, psrandom<DataType> *PRNG) : mvnormal(nullptr)
{
    if (PRNG)
    {
        if (!PRNG_initialized)
        {
            UMUQFAIL("One should set the state of the pseudo random number generator before setting it to this distribution!");
        }
        this->prng = PRNG;
        if (!this->prng->set_mvnormal(icovariance, n))
        {
            UMUQFAIL("Failed to set mvnormal object!");
        }
    }
    else
    {
        try
        {
            mvnormal.reset(new randomdist::multivariateNormalDistribution<DataType>(icovariance, n));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
multivariateGaussianDistribution<DataType, FunctionType>::multivariateGaussianDistribution(int const n, psrandom<DataType> *PRNG) : mvnormal(nullptr)
{
    if (PRNG)
    {
        if (!PRNG_initialized)
        {
            UMUQFAIL("One should set the state of the pseudo random number generator before setting it to this distribution!");
        }
        this->prng = PRNG;
        if (!this->prng->set_mvnormal(n))
        {
            UMUQFAIL("Failed to set mvnormal object!");
        }
    }
    else
    {
        try
        {
            mvnormal.reset(new randomdist::multivariateNormalDistribution<DataType>(n));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename DataType, class FunctionType>
multivariateGaussianDistribution<DataType, FunctionType>::~multivariateGaussianDistribution() {}

template <typename DataType, class FunctionType>
inline DataType multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_f(DataType const *x)
{
    if (this->prng)
    {
        DataType const denom = std::pow(M_2PI, this->numParams) * this->prng->mvnormal->lu.determinant();
        EVectorX<DataType> ax = EVectorMapTypeConst<DataType>(x, this->numParams) - this->prng->mvnormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        DataType const MDistSq = ax.transpose() * this->prng->mvnormal->lu.inverse() * ax;
        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }
    else
    {
        DataType const denom = std::pow(M_2PI, this->numParams) * mvnormal->lu.determinant();
        EVectorX<DataType> ax = EVectorMapTypeConst<DataType>(x, this->numParams) - mvnormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        DataType const MDistSq = ax.transpose() * mvnormal->lu.inverse() * ax;
        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }
}

template <typename DataType, class FunctionType>
inline DataType multivariateGaussianDistribution<DataType, FunctionType>::multivariategaussianDistribution_lf(DataType const *x)
{
    if (this->prng)
    {
        EVectorX<DataType> ax = EVectorMapTypeConst<DataType>(x, this->numParams) - this->prng->mvnormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        DataType const MDistSq = ax.transpose() * this->prng->mvnormal->lu.inverse() * ax;
        return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(this->prng->mvnormal->lu.determinant()));
    }
    else
    {
        EVectorX<DataType> ax = EVectorMapTypeConst<DataType>(x, this->numParams) - mvnormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        DataType const MDistSq = ax.transpose() * mvnormal->lu.inverse() * ax;
        return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(mvnormal->lu.determinant()));
    }
}

template <typename DataType, class FunctionType>
inline bool multivariateGaussianDistribution<DataType, FunctionType>::setRandomGenerator(psrandom<DataType> *PRNG)
{
    if (PRNG)
    {
        if (PRNG_initialized)
        {
            if (mvnormal)
            {
                this->prng = PRNG;
                this->prng->mvnormal = std::move(mvnormal);
                return true;
            }
            UMUQFAILRETURN("The pseudo-random number generator is already assigned!");
        }
        UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to this distribution!");
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename DataType, class FunctionType>
inline psrandom<DataType> *multivariateGaussianDistribution<DataType, FunctionType>::getRandomGenerator() { return this->prng; }

template <typename DataType, class FunctionType>
bool multivariateGaussianDistribution<DataType, FunctionType>::sample(DataType *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        EVectorMapType<DataType> X(x, this->numParams);
        X = this->prng->mvnormal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool multivariateGaussianDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        EVectorMapType<DataType> X(x.data(), this->numParams);
        X = this->prng->mvnormal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool multivariateGaussianDistribution<DataType, FunctionType>::sample(EVectorX<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        x = this->prng->mvnormal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool multivariateGaussianDistribution<DataType, FunctionType>::sample(DataType *x, int const nSamples)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        EMapType<DataType> X(x, nSamples, this->numParams);
        for (int i = 0; i < nSamples; i++)
        {
            X.row(i) = this->prng->mvnormal->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool multivariateGaussianDistribution<DataType, FunctionType>::sample(std::vector<DataType> &x, int const nSamples)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
#ifdef DEBUG
        if (nSamples * this->numParams > x.size())
        {
            UMUQFAILRETURN("The input size =", x.size(), " < requested samples size of ", nSamples * this->numParams, " !");
        }
#endif
        EMapType<DataType> X(x.data(), nSamples, this->numParams);
        for (int i = 0; i < nSamples; i++)
        {
            X.row(i) = this->prng->mvnormal->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

template <typename DataType, class FunctionType>
bool multivariateGaussianDistribution<DataType, FunctionType>::sample(EMatrixX<DataType> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        for (auto i = 0; i < x.cols(); i++)
        {
            x.col(i) = this->prng->mvnormal->dist();
        }
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

} // namespace density
} // namespace umuq

#endif // UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION
