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
 * \tparam RealType     Data type
 * \tparam FunctionType Function type
 * 
 * The Multivariate Gaussian Distribution is a generalization of the one-dimensional (univariate) Gaussian 
 * distribution to higher dimensions. One definition is that a random vector is said to be k-variate normally 
 * distributed if every linear combination of its k components has a univariate Gaussian distribution.
 * 
 * Reference:<br>
 * https://en.wikipedia.org/wiki/Multivariate_normal_distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x using mean vector Mean
 * and variance-covariance matrix Covariance. <br>
 * using:<br>
 * 
 * \f$
 * p(x_1,\cdots,x_k) = \frac{1}{\sqrt{(2 \pi)^k |\Sigma|}} \exp \left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right).
 * \f$
 * 
 * This class also provides random values x, distributed according to the Multivariate Gaussian Distribution probability 
 * density function. \sa sample
 */
template <typename RealType, class FunctionType = std::function<RealType(RealType const *)>>
class multivariateGaussianDistribution : public densityFunction<RealType, FunctionType>
{
  public:
    /*!
     * \brief Construct a new multivariateGaussianDistribution  object
     *
     * \param Mean        \f$ n \text{-dimensional}\f$ mean vector
     * \param Covariance  \f$ n \times n \f$ covariance matrix
     */
    multivariateGaussianDistribution(EVectorX<RealType> const &Mean, EMatrixX<RealType> const &Covariance);

    /*!
     * \brief Construct a new multivariateGaussianDistribution object
     * 
     * \param Mean        \f$ n \text{-dimensional}\f$ mean vector
     * \param Covariance  \f$ n \times n \f$ covariance matrix
     * \param n           Vector size (\f$n \text{-dimensional}\f$ vector)
     */
    multivariateGaussianDistribution(RealType const *Mean, RealType const *Covariance, int const n);

    /*!
     * \brief Construct a new multivariateGaussianDistribution object (default mean = 0)
     *
     * \param Covariance  \f$ n \times n \f$ covariance matrix
     */
    explicit multivariateGaussianDistribution(EMatrixX<RealType> const &Covariance);

    /*!
     * \brief Construct a new multivariateGaussianDistribution object (default mean = 0)
     * 
     * \param Covariance  \f$ n \times n \f$ covariance matrix
     * \param n           Vector size (\f$n \text{-dimensional}\f$ vector)
     */
    multivariateGaussianDistribution(RealType const *Covariance, int const n);

    /*!
     * \brief Construct a new multivariateGaussianDistribution object (default mean = 0, covariance=I)
     * 
     * \param n           Vector size (\f$n \text{-dimensional}\f$ vector)     
     */
    explicit multivariateGaussianDistribution(int const n);

    /*!
     * \brief Destroy the multinomial distribution object 
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
    inline RealType multivariategaussianDistribution_f(RealType const *x);

    /*!
     * \brief Log of multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param x  Input vector of size \f$ n \f$
     * 
     * \returns Log of density function value 
     */
    inline RealType multivariategaussianDistribution_lf(RealType const *x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Vector of samples
     */
    inline void sample(RealType *x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Vector of samples
     */
    inline void sample(std::vector<RealType> &x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Vector of samples
     */
    inline void sample(EVectorX<RealType> &x);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     */
    inline void sample(RealType *x, int const nSamples);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x         Vector of samples
     * \param nSamples  Number of sample vectors
     */
    inline void sample(std::vector<RealType> &x, int const nSamples);

    /*!
     * \brief Create samples of the Gaussian Distribution object
     *
     * \param x  Matrix of random samples 
     */
    inline void sample(EMatrixX<RealType> &x);

  private:
    /*! Multivariate random number distribution */
    std::unique_ptr<randomdist::multivariateNormalDistribution<RealType>> mvnormal;
};

template <typename RealType, class FunctionType>
multivariateGaussianDistribution<RealType, FunctionType>::multivariateGaussianDistribution(EVectorX<RealType> const &Mean, EMatrixX<RealType> const &Covariance) : mvnormal(nullptr)
{
    this->name = std::string("multivariategaussian");
    this->numParams = Mean.size();
    this->f = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(Mean, Covariance));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
multivariateGaussianDistribution<RealType, FunctionType>::multivariateGaussianDistribution(RealType const *Mean, RealType const *Covariance, int const n) : mvnormal(nullptr)
{
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(Mean, Covariance, n));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
multivariateGaussianDistribution<RealType, FunctionType>::multivariateGaussianDistribution(EMatrixX<RealType> const &Covariance) : mvnormal(nullptr)
{
    this->name = std::string("multivariategaussian");
    this->numParams = Covariance.rows();
    this->f = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(Covariance));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
multivariateGaussianDistribution<RealType, FunctionType>::multivariateGaussianDistribution(RealType const *Covariance, int const n) : mvnormal(nullptr)
{
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(Covariance, n));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
multivariateGaussianDistribution<RealType, FunctionType>::multivariateGaussianDistribution(int const n) : mvnormal(nullptr)
{
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(n));
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }
}

template <typename RealType, class FunctionType>
multivariateGaussianDistribution<RealType, FunctionType>::~multivariateGaussianDistribution() {}

template <typename RealType, class FunctionType>
inline RealType multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_f(RealType const *x)
{
    RealType const denom = std::pow(static_cast<RealType>(M_2PI), this->numParams) * mvnormal->lu.determinant();
    EVectorX<RealType> ax = EVectorMapTypeConst<RealType>(x, this->numParams) - mvnormal->mean;
    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    RealType const MDistSq = ax.transpose() * mvnormal->lu.inverse() * ax;
    return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
}

template <typename RealType, class FunctionType>
inline RealType multivariateGaussianDistribution<RealType, FunctionType>::multivariategaussianDistribution_lf(RealType const *x)
{
    EVectorX<RealType> ax = EVectorMapTypeConst<RealType>(x, this->numParams) - mvnormal->mean;
    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    RealType const MDistSq = ax.transpose() * mvnormal->lu.inverse() * ax;
    return -0.5 * (MDistSq + this->numParams * static_cast<RealType>(M_L2PI) + std::log(mvnormal->lu.determinant()));
}

template <typename RealType, class FunctionType>
inline void multivariateGaussianDistribution<RealType, FunctionType>::sample(RealType *x)
{
    EVectorMapType<RealType> X(x, this->numParams);
    X = mvnormal->dist();
}

template <typename RealType, class FunctionType>
inline void multivariateGaussianDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x)
{
    EVectorMapType<RealType> X(x.data(), this->numParams);
    X = mvnormal->dist();
}

template <typename RealType, class FunctionType>
inline void multivariateGaussianDistribution<RealType, FunctionType>::sample(EVectorX<RealType> &x)
{
    x = mvnormal->dist();
}

template <typename RealType, class FunctionType>
inline void multivariateGaussianDistribution<RealType, FunctionType>::sample(RealType *x, int const nSamples)
{
    mvnormal->dist(x, this->numParams, nSamples);
}

template <typename RealType, class FunctionType>
inline void multivariateGaussianDistribution<RealType, FunctionType>::sample(std::vector<RealType> &x, int const nSamples)
{
    mvnormal->dist(x.data(), this->numParams, nSamples);
}

template <typename RealType, class FunctionType>
inline void multivariateGaussianDistribution<RealType, FunctionType>::sample(EMatrixX<RealType> &x)
{
    mvnormal->dist(x);
}

} // namespace density
} // namespace umuq

#endif // UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION
