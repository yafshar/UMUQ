#ifndef UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION_H
#define UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION_H

namespace umuq
{

inline namespace density
{

/*! \class multivariategaussianDistribution
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
 * p(x_1,\cdots,x_k) = \frac{1}{\sqrt{(2 \pi)^k |\Sigma|}} \exp \left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right).
 * \f$
 * 
 * It also provides random values x, distributed according to the Multivariate Gaussian Distribution probability 
 * density function. 
 * 
 * \note
 * - For using any member function, a pointer to a Random Number Generator object \sa psrandom in the 
 * construction is required, otherwise, it fails.
 * 
 * \tparam T Data type
 */
template <typename T, class V = T const *>
class multivariategaussianDistribution : public densityFunction<T, std::function<T(V)>>
{
  public:
    /*!
     * \brief Construct a new multivariategaussian distribution object
     *
     * \param imean        Mean vector of size \f$n\f$
     * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
     * \param PRNG         Pseudo-random number object \sa psrandom
     */
    multivariategaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object
     * 
     * \param imean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     * \param PRNG         Pseudo-random number object \sa psrandom
     */
    multivariategaussianDistribution(T const *imean, T const *icovariance, int const n, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param PRNG         Pseudo-random number object \sa psrandom
     */
    explicit multivariategaussianDistribution(EMatrixX<T> const &icovariance, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     * 
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     * \param PRNG         Pseudo-random number object \sa psrandom
     */
    multivariategaussianDistribution(T const *icovariance, int const n, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0, covariance=I)
     * 
     * \param n vector size
     * \param PRNG     Pseudo-random number object \sa psrandom
     */
    explicit multivariategaussianDistribution(int const n, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Destroy the multinomial distribution object
     * 
     */
    ~multivariategaussianDistribution();

    /*!
     * \brief Multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param x  Input vector of size \f$ n \f$
     * 
     * \returns Density function value 
     */
    inline T multivariategaussianDistribution_f(T const *x);

    /*!
     * \brief Log of multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param x  Input vector of size \f$ n \f$
     * 
     * \returns  Log of density function value 
     */
    inline T multivariategaussianDistribution_lf(T const *x);

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

  private:
    /*!
     * \brief Multivariate random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::multivariatenormalDistribution<T>> mvnormal;
};

/*! \class multivariateGaussianDistribution
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
 * p(x_1,\cdots,x_k) = \frac{1}{\sqrt{(2 \pi)^k |\Sigma|}} \exp \left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right).
 * \f$
 * 
 * It also provides random values x, distributed according to the Multivariate Gaussian Distribution probability 
 * density function. 
 * 
 * 
 * \tparam T Data type
 */
template <typename T, class V = T const *>
class multivariateGaussianDistribution : public densityFunction<T, std::function<T(V)>>
{
  public:
    /*!
     * \brief Construct a new multivariategaussian distribution object
     *
     * \param imean        Mean vector of size \f$n\f$
     * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
     */
    multivariateGaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object
     * 
     * \param imean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     */
    multivariateGaussianDistribution(T const *imean, T const *icovariance, int const n, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     */
    explicit multivariateGaussianDistribution(EMatrixX<T> const &icovariance, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     * 
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     */
    multivariateGaussianDistribution(T const *icovariance, int const n, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0, covariance=I)
     * 
     * \param n vector size
     */
    explicit multivariateGaussianDistribution(int const n, psrandom<T> *PRNG = nullptr);

    /*!
     * \brief Destroy the multinomial distribution object
     * 
     */
    ~multivariateGaussianDistribution() {}

    /*!
     * \brief Multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param x  Input vector of size \f$ n \f$
     * 
     * \returns Density function value 
     */
    inline T multivariateGaussianDistribution_f(T const *x);

    /*!
     * \brief Log of multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param x  Input vector of size \f$ n \f$
     * 
     * \returns  Log of density function value 
     */
    inline T multivariateGaussianDistribution_lf(T const *x);

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
     * \brief Create samples of the Multivariate Gaussian distribution
     *
     * \param x  Vector of samples
     *
     * \return true
     * \return false If Random Number Generator object is not assigned
     */
    bool sample(T *x);
    bool sample(std::vector<T> &x);

  private:
    /*!
     * \brief Multivariate random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::multivariateNormalDistribution<T>> mvNormal;
};

template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance, psrandom<T> *PRNG) : mvnormal(nullptr)
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
            mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(imean, icovariance));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = imean.size();
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(T const *imean, T const *icovariance, int const n, psrandom<T> *PRNG) : mvnormal(nullptr)
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
            mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(imean, icovariance, n));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(EMatrixX<T> const &icovariance, psrandom<T> *PRNG) : mvnormal(nullptr)
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
            mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(icovariance));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = icovariance.rows();
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(T const *icovariance, int const n, psrandom<T> *PRNG) : mvnormal(nullptr)
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
            mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(icovariance, n));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(int const n, psrandom<T> *PRNG) : mvnormal(nullptr)
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
            mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(n));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
multivariategaussianDistribution<T, V>::~multivariategaussianDistribution() {}

template <typename T, class V>
inline T multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f(T const *x)
{
    if (this->prng)
    {
        T const denom = std::pow(M_2PI, this->numParams) * this->prng->mvnormal->lu.determinant();
        EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - this->prng->mvnormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T const MDistSq = ax.transpose() * this->prng->mvnormal->lu.inverse() * ax;
        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }
    else
    {
        T const denom = std::pow(M_2PI, this->numParams) * mvnormal->lu.determinant();
        EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - mvnormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T const MDistSq = ax.transpose() * mvnormal->lu.inverse() * ax;
        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }
}

template <typename T, class V>
inline T multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf(T const *x)
{
    if (this->prng)
    {
        EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - this->prng->mvnormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T const MDistSq = ax.transpose() * this->prng->mvnormal->lu.inverse() * ax;
        return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(this->prng->mvnormal->lu.determinant()));
    }
    else
    {
        EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - mvnormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T const MDistSq = ax.transpose() * mvnormal->lu.inverse() * ax;
        return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(mvnormal->lu.determinant()));
    }
}

template <typename T, class V>
inline bool multivariategaussianDistribution<T, V>::setRandomGenerator(psrandom<T> *PRNG)
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

template <typename T, class V>
bool multivariategaussianDistribution<T, V>::sample(T *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        EVectorMapType<T> X(x, this->numParams);
        X = this->prng->mvnormal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!")
#endif
}

template <typename T, class V>
bool multivariategaussianDistribution<T, V>::sample(std::vector<T> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        EVectorMapType<T> X(x.data(), this->numParams);
        X = this->prng->mvnormal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!")
#endif
}

template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance, psrandom<T> *PRNG) : mvNormal(nullptr)
{
    if (PRNG)
    {
        this->prng = PRNG;
        if (!this->prng->set_mvNormal(imean, icovariance))
        {
            UMUQFAIL("Failed to set mvNormal object!");
        }
    }
    else
    {
        try
        {
            mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(imean, icovariance));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = imean.size();
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(T const *imean, T const *icovariance, int const n, psrandom<T> *PRNG) : mvNormal(nullptr)
{
    if (PRNG)
    {
        this->prng = PRNG;
        if (!this->prng->set_mvNormal(imean, icovariance, n))
        {
            UMUQFAIL("Failed to set mvNormal object!");
        }
    }
    else
    {
        try
        {
            mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(imean, icovariance, n));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(EMatrixX<T> const &icovariance, psrandom<T> *PRNG) : mvNormal(nullptr)
{
    if (PRNG)
    {
        this->prng = PRNG;
        if (!this->prng->set_mvNormal(icovariance))
        {
            UMUQFAIL("Failed to set mvNormal object!");
        }
    }
    else
    {
        try
        {
            mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(icovariance));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = icovariance.rows();
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(T const *icovariance, int const n, psrandom<T> *PRNG) : mvNormal(nullptr)
{
    if (PRNG)
    {
        this->prng = PRNG;
        if (!this->prng->set_mvNormal(icovariance, n))
        {
            UMUQFAIL("Failed to set mvNormal object!");
        }
    }
    else
    {
        try
        {
            mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(icovariance, n));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(int const n, psrandom<T> *PRNG) : mvNormal(nullptr)
{
    if (PRNG)
    {
        this->prng = PRNG;
        if (!this->prng->set_mvNormal(n))
        {
            UMUQFAIL("Failed to set mvNormal object!");
        }
    }
    else
    {
        try
        {
            mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(n));
        }
        catch (...)
        {
            UMUQFAIL("Failed to allocate memory!");
        }
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

template <typename T, class V>
inline T multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f(T const *x)
{
    if (this->prng)
    {
        T const denom = std::pow(M_2PI, this->numParams) * this->prng->mvNormal->lu.determinant();
        EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - this->prng->mvNormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T const MDistSq = ax.transpose() * this->prng->mvNormal->lu.inverse() * ax;
        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }
    else
    {
        T const denom = std::pow(M_2PI, this->numParams) * mvNormal->lu.determinant();
        EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - mvNormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T const MDistSq = ax.transpose() * mvNormal->lu.inverse() * ax;
        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }
}

template <typename T, class V>
inline T multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf(T const *x)
{
    if (this->prng)
    {
        EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - this->prng->mvNormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T MDistSq = ax.transpose() * this->prng->mvNormal->lu.inverse() * ax;
        return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(this->prng->mvNormal->lu.determinant()));
    }
    else
    {
        EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - mvNormal->mean;
        // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T MDistSq = ax.transpose() * mvNormal->lu.inverse() * ax;
        return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(mvNormal->lu.determinant()));
    }
}

template <typename T, class V>
inline bool multivariateGaussianDistribution<T, V>::setRandomGenerator(psrandom<T> *PRNG)
{
    if (PRNG)
    {
        if (mvNormal)
        {
            this->prng = PRNG;
            this->prng->mvNormal = std::move(mvNormal);
            return true;
        }
        UMUQFAILRETURN("The pseudo-random number generator is already assigned!");
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename T, class V>
bool multivariateGaussianDistribution<T, V>::sample(T *x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        EVectorMapType<T> X(x, this->numParams);
        X = this->prng->mvNormal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!")
#endif
}

template <typename T, class V>
bool multivariateGaussianDistribution<T, V>::sample(std::vector<T> &x)
{
#ifdef DEBUG
    if (this->prng)
    {
#endif
        EVectorMapType<T> X(x.data(), this->numParams);
        X = this->prng->mvNormal->dist();
        return true;
#ifdef DEBUG
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!")
#endif
}

} // namespace density
} // namespace umuq

#endif // UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION_H
