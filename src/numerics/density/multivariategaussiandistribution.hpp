#ifndef UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION_H
#define UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION_H

#include "../function/densityfunction.hpp"

/*! \class multivariategaussianDistribution
 * \brief The Multivariate Gaussian Distribution
 * 
 * \tparam T Data type
 * 
 * The Multivariate Gaussian Distribution is a generalization of the one-dimensional (univariate) Gaussian 
 * distribution to higher dimensions. One definition is that a random vector is said to be k-variate normally 
 * distributed if every linear combination of its k components has a univariate Gaussian distribution.
 * 
 * Reference:
 * https://en.wikipedia.org/wiki/Multivariate_normal_distribution
 * 
 *
 * This class provides probability density \f$ p(x) \f$ and it's Log at x using mean vector imean
 * and variance-covariance matrix icovariance.
 * using: 
 * \f[
 * p(x_1,\cdots,x_k) = \frac{1}{\sqrt{(2 \pi)^k |\Sigma|}} \exp \left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right).
 * \f]
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
     */
    multivariategaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance);

    /*!
     * \brief Construct a new multivariategaussian distribution object
     * 
     * \param imean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     */
    multivariategaussianDistribution(T const *imean, T const *icovariance, int const n);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     */
    explicit multivariategaussianDistribution(EMatrixX<T> const &icovariance);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     * 
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     */
    multivariategaussianDistribution(T const *icovariance, int const n);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0, covariance=I)
     * 
     * \param n vector size
     */
    explicit multivariategaussianDistribution(int const n);

    /*!
     * \brief Destroy the multinomial distribution object
     * 
     */
    ~multivariategaussianDistribution() {}

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

  private:
    //! Psuedo Random number
    psrandom<T> prng;
};

/*!
 * \brief Construct a new multivariategaussian distribution object
 *
 * \param imean        Mean vector of size \f$n\f$
 * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance)
{
    if (!this->prng.set_mvnormal(imean, icovariance))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = imean.size();
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Construct a new multivariategaussian distribution object
 * 
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(T const *imean, T const *icovariance, int const n)
{
    if (!this->prng.set_mvnormal(imean, icovariance, n))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(EMatrixX<T> const &icovariance)
{
    if (!this->prng.set_mvnormal(icovariance))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = icovariance.rows();
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0)
 * 
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(T const *icovariance, int const n)
{
    if (!this->prng.set_mvnormal(icovariance, n))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0, covariance=I)
 * 
 * \param n vector size
 */
template <typename T, class V>
multivariategaussianDistribution<T, V>::multivariategaussianDistribution(int const n)
{
    if (!this->prng.set_mvnormal(n))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param x  Input vector of size \f$ n \f$
 * 
 * \returns Density function value 
 */
template <typename T, class V>
inline T multivariategaussianDistribution<T, V>::multivariategaussianDistribution_f(T const *x)
{
    T denom = std::pow(M_2PI, this->numParams) * this->prng.mvnormal->lu.determinant();

    EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - this->prng.mvnormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    T MDistSq = ax.transpose() * this->prng.mvnormal->lu.inverse() * ax;

    return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
}

/*!
 * \brief Log of multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param x  Input vector of size \f$ n \f$
 * 
 * \returns  Log of density function value 
 */
template <typename T, class V>
inline T multivariategaussianDistribution<T, V>::multivariategaussianDistribution_lf(T const *x)
{
    EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - this->prng.mvnormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    T MDistSq = ax.transpose() * this->prng.mvnormal->lu.inverse() * ax;

    return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(this->prng.mvnormal->lu.determinant()));
}

/*! \class multivariateGaussianDistribution
 * \brief The Multivariate Gaussian Distribution
 * 
 * The Multivariate Gaussian Distribution is a generalization of the one-dimensional (univariate) Gaussian 
 * distribution to higher dimensions. One definition is that a random vector is said to be k-variate normally 
 * distributed if every linear combination of its k components has a univariate Gaussian distribution.
 * 
 * Reference:
 * https://en.wikipedia.org/wiki/Multivariate_normal_distribution
 * 
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x using mean vector imean
 * and variance-covariance matrix icovariance.
 * using: 
 * \f[
 * p(x_1,\cdots,x_k) = \frac{1}{\sqrt{(2 \pi)^k |\Sigma|}} \exp \left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right).
 * \f]
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
    multivariateGaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance);

    /*!
     * \brief Construct a new multivariategaussian distribution object
     * 
     * \param imean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     */
    multivariateGaussianDistribution(T const *imean, T const *icovariance, int const n);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     */
    explicit multivariateGaussianDistribution(EMatrixX<T> const &icovariance);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0)
     * 
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     */
    multivariateGaussianDistribution(T const *icovariance, int const n);

    /*!
     * \brief Construct a new multivariategaussian distribution object (default mean = 0, covariance=I)
     * 
     * \param n vector size
     */
    explicit multivariateGaussianDistribution(int const n);

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

  private:
    //! Psuedo Random number
    psrandom<T> prng;
};

/*!
 * \brief Construct a new multivariategaussian distribution object
 *
 * \param imean        Mean vector of size \f$n\f$
 * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance)
{
    if (!this->prng.set_mvNormal(imean, icovariance))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = imean.size();
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Construct a new multivariategaussian distribution object
 * 
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(T const *imean, T const *icovariance, int const n)
{
    if (!this->prng.set_mvNormal(imean, icovariance, n))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(EMatrixX<T> const &icovariance)
{
    if (!this->prng.set_mvNormal(icovariance))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = icovariance.rows();
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0)
 * 
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(T const *icovariance, int const n)
{
    if (!this->prng.set_mvNormal(icovariance, n))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0, covariance=I)
 * 
 * \param n vector size
 */
template <typename T, class V>
multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution(int const n)
{
    if (!this->prng.set_mvNormal(n))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
    this->f = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param x  Input vector of size \f$ n \f$
 * 
 * \returns Density function value 
 */
template <typename T, class V>
inline T multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_f(T const *x)
{
    T denom = std::pow(M_2PI, this->numParams) * this->prng.mvNormal->lu.determinant();

    EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - this->prng.mvNormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    T MDistSq = ax.transpose() * this->prng.mvNormal->lu.inverse() * ax;

    return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
}

/*!
 * \brief Log of multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param x  Input vector of size \f$ n \f$
 * 
 * \returns  Log of density function value 
 */
template <typename T, class V>
inline T multivariateGaussianDistribution<T, V>::multivariateGaussianDistribution_lf(T const *x)
{
    EVectorX<T> ax = EVectorMapTypeConst<T>(x, this->numParams) - this->prng.mvNormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    T MDistSq = ax.transpose() * this->prng.mvNormal->lu.inverse() * ax;

    return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(this->prng.mvNormal->lu.determinant()));
}

#endif // UMUQ_MULTIVARIATEGAUSSIANDISTRIBUTION_H
