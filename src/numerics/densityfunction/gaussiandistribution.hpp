#ifndef UMUQ_GAUSSIANDISTRIBUTION_H
#define UMUQ_GAUSSIANDISTRIBUTION_H

#include "densityfunction.hpp"

/*! \class gaussianDistribution
 * \brief The Gaussian distribution
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a Gaussian 
 * distribution with standard deviation \f$ \sigma \f$
 * using: 
 * \f[
 * p(x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(-\frac{\left(x - \mu \right)^2}{2\sigma^2}\right)}
 * \f]
 * 
 * \tparam T Data type
 */
template <typename T>
class gaussianDistribution : public densityFunction<T, gaussianDistribution<T>>
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
     * \brief Destroy the gaussian Distribution object
     *  
     */
    ~gaussianDistribution() {}

    /*!
     * \brief Gaussian Distribution density function
     * 
     * \param x Input value
     * 
     * \returns Density function value 
     */
    inline T f(T const x);

    /*!
     * \brief Log of Gaussian Distribution density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T lf(T const x);
};

/*!
 * \brief Construct a new gaussian Distribution object
 * 
 * \param mu     Mean, \f$ \mu \f$
 * \param sigma  Standard deviation \f$ \sigma \f$
 */
template <typename T>
gaussianDistribution<T>::gaussianDistribution(T const mu, T const sigma) : densityFunction<T, gaussianDistribution<T>>(std::vector<T>{mu, sigma}.data(), 2, "gaussian") {}
/*!
 * \brief Gaussian Distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T>
inline T gaussianDistribution<T>::f(T const x)
{
    T const xSigma = x - this->params[0] / this->params[1];
    return static_cast<T>(1) / (std::sqrt(M_2PI) * this->params[1]) * std::exp(-xSigma * xSigma / static_cast<T>(2));
}

/*!
 * \brief Log of Gaussian Distribution density function
 * 
 * \param x Input value
 * 
 * \returns  Log of density function value 
 */
template <typename T>
inline T gaussianDistribution<T>::lf(T const x)
{
    T const xSigma = x - this->params[0] / this->params[1];
    return -0.5 * M_L2PI - std::log(this->params[1]) - 0.5 * xSigma * xSigma;
}

/*! \class multivariategaussianDistribution
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
template <typename T>
class multivariategaussianDistribution : public densityFunction<T, multivariategaussianDistribution<T>>
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
    inline T f(T const *x);

    /*!
     * \brief Multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param X  Input vector of size \f$ n \f$
     * 
     * \returns Density function value 
     */
    inline T f(EVectorX<T> const &X);

    /*!
     * \brief Log of multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param x  Input vector of size \f$ n \f$
     * 
     * \returns  Log of density function value 
     */
    inline T lf(T const *x);

    /*!
     * \brief Log of multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param X  Input vector of size \f$ n \f$
     * 
     * \returns  Log of density function value 
     */
    inline T lf(EVectorX<T> const &X);

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
template <typename T>
multivariategaussianDistribution<T>::multivariategaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance)
{
    if (!this->prng.set_mvnormal(imean, icovariance))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = imean.size();
}

/*!
 * \brief Construct a new multivariategaussian distribution object
 * 
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
multivariategaussianDistribution<T>::multivariategaussianDistribution(T const *imean, T const *icovariance, int const n)
{
    if (!this->prng.set_mvnormal(imean, icovariance, n))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
explicit multivariategaussianDistribution<T>::multivariategaussianDistribution(EMatrixX<T> const &icovariance)
{
    if (!this->prng.set_mvnormal(icovariance))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = icovariance.rows();
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0)
 * 
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
multivariategaussianDistribution<T>::multivariategaussianDistribution(T const *icovariance, int const n)
{
    if (!this->prng.set_mvnormal(icovariance, n))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0, covariance=I)
 * 
 * \param n vector size
 */
template <typename T>
explicit multivariategaussianDistribution<T>::multivariategaussianDistribution(int const n)
{
    if (!this->prng.set_mvnormal(n))
    {
        UMUQFAIL("Failed to set mvnormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
}

/*!
 * \brief Multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param x  Input vector of size \f$ n \f$
 * 
 * \returns Density function value 
 */
template <typename T>
inline T multivariategaussianDistribution<T>::f(T const *x)
{
    CTEMapX<T, Eigen::ColMajor> X(x, this->numParams, 1);

    T denom = std::pow(M_2PI, this->numParams) * this->prng.mvnormal->lu.determinant();

    EVectorX<T> ax = X - this->prng.mvnormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    T MDistSq = ax.transpose() * this->prng.mvnormal->lu.inverse() * ax;

    return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
}

/*!
 * \brief Multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param X  Input vector of size \f$ n \f$
 * 
 * \returns Density function value 
 */
template <typename T>
inline T multivariategaussianDistribution<T>::f(EVectorX<T> const &X)
{
    T denom = std::pow(M_2PI, this->numParams) * this->prng.mvnormal->lu.determinant();

    EVectorX<T> ax = X - this->prng.mvnormal->mean;

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
template <typename T>
inline T multivariategaussianDistribution<T>::lf(T const *x)
{
    CTEMapX<T, Eigen::ColMajor> X(x, this->numParams, 1);

    EVectorX<T> ax = X - this->prng.mvnormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    T MDistSq = ax.transpose() * this->prng.mvnormal->lu.inverse() * ax;

    return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(this->prng.mvnormal->lu.determinant()));
}

/*!
 * \brief Log of multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param X  Input vector of size \f$ n \f$
 * 
 * \returns  Log of density function value 
 */
template <typename T>
inline T multivariategaussianDistribution<T>::lf(EVectorX<T> const &X)
{
    EVectorX<T> ax = X - this->prng.mvnormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$\ mu \f$
    T MDistSq = ax.transpose() * this->prng.mvnormal->lu.inverse() * ax;

    return -0.5 * (MDistSq + X.rows() * M_L2PI + std::log(this->prng.mvnormal->lu.determinant()));
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
template <typename T>
class multivariateGaussianDistribution : public densityFunction<T, multivariateGaussianDistribution<T>>
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
    inline T f(T const *x);

    /*!
     * \brief Multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param X  Input vector of size \f$ n \f$
     * 
     * \returns Density function value 
     */
    inline T f(EVectorX<T> const &X);

    /*!
     * \brief Log of multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param x  Input vector of size \f$ n \f$
     * 
     * \returns  Log of density function value 
     */
    inline T lf(T const *x);

    /*!
     * \brief Log of multivariate Gaussian distribution density function
     * Computes the probability from the Multivariate Gaussian distribution
     * 
     * \param X  Input vector of size \f$ n \f$
     * 
     * \returns  Log of density function value 
     */
    inline T lf(EVectorX<T> const &X);

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
template <typename T>
multivariateGaussianDistribution<T>::multivariateGaussianDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance)
{
    if (!this->prng.set_mvNormal(imean, icovariance))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = imean.size();
}

/*!
 * \brief Construct a new multivariategaussian distribution object
 * 
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
multivariateGaussianDistribution<T>::multivariateGaussianDistribution(T const *imean, T const *icovariance, int const n)
{
    if (!this->prng.set_mvNormal(imean, icovariance, n))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
explicit multivariateGaussianDistribution<T>::multivariateGaussianDistribution(EMatrixX<T> const &icovariance)
{
    if (!this->prng.set_mvNormal(icovariance))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = icovariance.rows();
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0)
 * 
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
multivariateGaussianDistribution<T>::multivariateGaussianDistribution(T const *icovariance, int const n)
{
    if (!this->prng.set_mvNormal(icovariance, n))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
}

/*!
 * \brief Construct a new multivariategaussian distribution object (default mean = 0, covariance=I)
 * 
 * \param n vector size
 */
template <typename T>
explicit multivariateGaussianDistribution<T>::multivariateGaussianDistribution(int const n)
{
    if (!this->prng.set_mvNormal(n))
    {
        UMUQFAIL("Failed to set mvNormal object!");
    }
    this->name = std::string("multivariategaussian");
    this->numParams = n;
}

/*!
 * \brief Multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param x  Input vector of size \f$ n \f$
 * 
 * \returns Density function value 
 */
template <typename T>
inline T multivariateGaussianDistribution<T>::f(T const *x)
{
    CTEMapX<T, Eigen::ColMajor> X(x, this->numParams, 1);

    T denom = std::pow(M_2PI, this->numParams) * this->prng.mvNormal->lu.determinant();

    EVectorX<T> ax = X - this->prng.mvNormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    T MDistSq = ax.transpose() * this->prng.mvNormal->lu.inverse() * ax;

    return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
}

/*!
 * \brief Multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param X  Input vector of size \f$ n \f$
 * 
 * \returns Density function value 
 */
template <typename T>
inline T multivariateGaussianDistribution<T>::f(EVectorX<T> const &X)
{
    T denom = std::pow(M_2PI, X.rows()) * this->prng.mvNormal->lu.determinant();

    EVectorX<T> ax = X - this->prng.mvNormal->mean;

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
template <typename T>
inline T multivariateGaussianDistribution<T>::lf(T const *x)
{
    CTEMapX<T, Eigen::ColMajor> X(x, this->numParams, 1);

    EVectorX<T> ax = X - this->prng.mvNormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
    T MDistSq = ax.transpose() * this->prng.mvNormal->lu.inverse() * ax;

    return -0.5 * (MDistSq + this->numParams * M_L2PI + std::log(this->prng.mvNormal->lu.determinant()));
}

/*!
 * \brief Log of multivariate Gaussian distribution density function
 * Computes the probability from the Multivariate Gaussian distribution
 * 
 * \param X  Input vector of size \f$ n \f$
 * 
 * \returns  Log of density function value 
 */
template <typename T>
inline T multivariateGaussianDistribution<T>::lf(EVectorX<T> const &X)
{
    EVectorX<T> ax = X - this->prng.mvNormal->mean;

    // Mahalanobis distance between \f$ X \f$ and \f$\ mu \f$
    T MDistSq = ax.transpose() * this->prng.mvNormal->lu.inverse() * ax;

    return -0.5 * (MDistSq + X.rows() * M_L2PI + std::log(this->prng.mvNormal->lu.determinant()));
}

#endif // UMUQ_GAUSSIANDISTRIBUTION_H
