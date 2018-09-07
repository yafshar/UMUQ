#ifndef UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION_H
#define UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION_H

namespace umuq
{
namespace randomdist
{

/*! \class multivariatenormalDistribution
 * \brief Multivariate normal distribution
 * 
 * NOTE: This should be called after setting the State of psrandom object
 * 
 */
template <typename T = double>
class multivariatenormalDistribution
{
public:
  /*!
   * \brief constructor
   *
   * \param imean        Mean vector of size \f$n\f$
   * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
   */
  multivariatenormalDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance);

  /*!
   * \brief constructor
   * 
   * \param imean        Input mean vector of size \f$n\f$
   * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n            Vector size
   */
  multivariatenormalDistribution(T const *imean, T const *icovariance, int const n);

  /*!
   * \brief constructor (default mean = 0)
   *
   * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
   */
  explicit multivariatenormalDistribution(EMatrixX<T> const &icovariance);

  /*!
   * \brief constructor (default mean = 0)
   * 
   * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n            Vector size
   */
  multivariatenormalDistribution(T const *icovariance, int const n);

  /*!
   * \brief constructor (default mean = 0, covariance=I)
   * 
   * \param n vector size
   */
  explicit multivariatenormalDistribution(int const n);

  /*!
   * \brief Move constructor, construct a new multivariatenormalDistribution object from input multivariatenormalDistribution object
   * 
   * \param other  Input multivariatenormalDistribution object
   */
  multivariatenormalDistribution(multivariatenormalDistribution<T> &&other);

  /*!
   * \brief Move assignment operator
   * 
   * \param other  Input multivariatenormalDistribution object
   * \return multivariatenormalDistribution& 
   */
  multivariatenormalDistribution<T> &operator=(multivariatenormalDistribution<T> &&other);

  /*!
   * \returns a vector with multivariate normal distribution
   */
  EVectorX<T> operator()();

  /*!
   * \returns a vector with multivariate normal distribution
   */
  EVectorX<T> dist();

private:
  // Make it noncopyable
  multivariatenormalDistribution(multivariatenormalDistribution<T> const &) = delete;

  // Make it not assignable
  multivariatenormalDistribution<T> &operator=(multivariatenormalDistribution<T> const &) = delete;

public:
  //! Vector of size \f$n\f$
  EVectorX<T> mean;

  //! Variance-covariance matrix of size \f$ n \times n \f$
  EMatrixX<T> covariance;

private:
  //! Matrix of size \f$n \times n\f$
  EMatrixX<T> transform;

public:
  //! LU decomposition of a matrix with complete pivoting
  Eigen::FullPivLU<EMatrixX<T>> lu;

private:
  //! Generates random numbers according to the Normal (or Gaussian) random number distribution
  std::normal_distribution<T> d;
};

/*! \class multivariatenormalDistribution
 * \brief Multivariate normal distribution
 * 
 * NOTE: This can be called without setting the State of psrandom object
 * 
 */
template <typename T = double>
class multivariateNormalDistribution
{
public:
  /*!
   * \brief constructor
   *
   * \param imean        Mean vector of size \f$n\f$
   * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
   */
  multivariateNormalDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance);

  /*!
   * \brief constructor
   * 
   * \param imean        Input mean vector of size \f$n\f$
   * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n            Vector size
   */
  multivariateNormalDistribution(T const *imean, T const *icovariance, int const n);

  /*!
   * \brief constructor (default mean = 0)
   *
   * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
   */
  explicit multivariateNormalDistribution(EMatrixX<T> const &icovariance);

  /*!
   * \brief constructor (default mean = 0)
   * 
   * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n            Vector size
   */
  multivariateNormalDistribution(T const *icovariance, int const n);

  /*!
   * \brief constructor (default mean = 0, covariance=I)
   * 
   * \param n vector size
   */
  explicit multivariateNormalDistribution(int const n);

  /*!
   * \brief Move constructor, construct a new multivariateNormalDistribution object from input multivariateNormalDistribution object
   * 
   * \param other  Input multivariateNormalDistribution object
   */
  multivariateNormalDistribution(multivariateNormalDistribution<T> &&other);

  /*!
   * \brief Move assignment operator
   * 
   * \param other  Input multivariateNormalDistribution object
   * \return multivariateNormalDistribution& 
   */
  multivariateNormalDistribution<T> &operator=(multivariateNormalDistribution<T> &&other);

  /*!
   * \returns a vector with multivariate normal distribution
   */
  EVectorX<T> operator()();

  /*!
   * \returns a vector with multivariate normal distribution
   */
  EVectorX<T> dist();

private:
  // Make it noncopyable
  multivariateNormalDistribution(multivariateNormalDistribution<T> const &) = delete;

  // Make it not assignable
  multivariateNormalDistribution<T> &operator=(multivariateNormalDistribution<T> const &) = delete;

public:
  //! Vector of size \f$n\f$
  EVectorX<T> mean;

  //! Variance-covariance matrix of size \f$n \times n\f$
  EMatrixX<T> covariance;

private:
  //! Matrix of size \f$n \times n\f$
  EMatrixX<T> transform;

public:
  //! LU decomposition of a matrix with complete pivoting
  Eigen::FullPivLU<EMatrixX<T>> lu;

private:
  //! A random number engine based on Mersenne Twister algorithm
  std::mt19937 gen;

  //! Generates random numbers according to the Normal (or Gaussian) random number distribution
  std::normal_distribution<T> d;
};

/*!
 * \brief constructor
 *
 * \param imean        Mean vector of size \f$n\f$
 * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance) : mean(imean),
                                                                                                                              covariance(icovariance),
                                                                                                                              lu(icovariance)
{
  if (!PRNG_initialized)
  {
    UMUQFAIL("One should set the current state of the engine before constructing this object!");
  }

  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor
 * 
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(T const *imean, T const *icovariance, int const n) : mean(EMapTypeConst<T>(imean, n, 1)),
                                                                                                                       covariance(EMapTypeConst<T>(icovariance, n, n)),
                                                                                                                       lu(EMapTypeConst<T>(icovariance, n, n))
{
  if (!PRNG_initialized)
  {
    UMUQFAIL("One should set the current state of the engine before constructing this object!");
  }

  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(EMatrixX<T> const &icovariance) : multivariatenormalDistribution(EVectorX<T>::Zero(icovariance.rows()), icovariance)
{
  if (!PRNG_initialized)
  {
    UMUQFAIL("One should set the current state of the engine before constructing this object!");
  }
}

/*!
 * \brief constructor (default mean = 0)
 * 
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(T const *icovariance, int const n) : mean(EVectorX<T>::Zero(n)),
                                                                                                       covariance(EMapTypeConst<T>(icovariance, n, n)),
                                                                                                       lu(EMapTypeConst<T>(icovariance, n, n))
{
  if (!PRNG_initialized)
  {
    UMUQFAIL("One should set the current state of the engine before constructing this object!");
  }

  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor (default mean = 0, covariance=I)
 * 
 * \param n vector size
 */
template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(int const n) : mean(EVectorX<T>::Zero(n)),
                                                                                 covariance(EMatrixX<T>::Identity(n, n)),
                                                                                 transform(EMatrixX<T>::Identity(n, n)),
                                                                                 lu(EMatrixX<T>::Identity(n, n))
{
  if (!PRNG_initialized)
  {
    UMUQFAIL("One should set the current state of the engine before constructing this object!");
  }
}

/*!
 * \brief Move constructor, construct a new multivariatenormalDistribution object from input multivariatenormalDistribution object
 * 
 * \param other  Input multivariatenormalDistribution object
 */
template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(multivariatenormalDistribution<T> &&other) : mean(std::move(other.mean)),
                                                                                                               covariance(std::move(other.covariance)),
                                                                                                               transform(std::move(other.transform)),
                                                                                                               lu(std::move(other.lu)),
                                                                                                               d(std::move(other.d)) {}

/*!
 * \brief Move assignment operator
 * 
 * \param other  Input multivariatenormalDistribution object
 * \return multivariatenormalDistribution& 
 */
template <typename T>
multivariatenormalDistribution<T> &multivariatenormalDistribution<T>::operator=(multivariatenormalDistribution<T> &&other)
{
  mean = std::move(other.mean);
  covariance = std::move(other.covariance);
  transform = std::move(other.transform);
  lu = std::move(other.lu);
  d = std::move(other.d);
  return *this;
}

/*!
 * \returns a vector with multivariate normal distribution
 */
template <typename T>
EVectorX<T> multivariatenormalDistribution<T>::operator()()
{
  int const me = torc_i_worker_id();
  return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T const x) { return d(NumberGenerator[me]); });
}

/*!
 * \returns a vector with multivariate normal distribution
 */
template <typename T>
EVectorX<T> multivariatenormalDistribution<T>::dist()
{
  int const me = torc_i_worker_id();
  return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T const x) { return d(NumberGenerator[me]); });
}

/*!
 * \brief constructor
 *
 * \param imean        Mean vector of size \f$n\f$
 * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance) : mean(imean),
                                                                                                                              covariance(icovariance),
                                                                                                                              lu(icovariance),
                                                                                                                              gen(std::random_device{}())
{
  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor
 * 
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(T const *imean, T const *icovariance, int const n) : mean(EMapTypeConst<T>(imean, n, 1)),
                                                                                                                       covariance(EMapTypeConst<T>(icovariance, n, n)),
                                                                                                                       lu(EMapTypeConst<T>(icovariance, n, n)),
                                                                                                                       gen(std::random_device{}())
{
  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(EMatrixX<T> const &icovariance) : multivariateNormalDistribution(EVectorX<T>::Zero(icovariance.rows()), icovariance) {}

/*!
 * \brief constructor (default mean = 0)
 * 
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(T const *icovariance, int const n) : mean(EVectorX<T>::Zero(n)),
                                                                                                       covariance(EMapTypeConst<T>(icovariance, n, n)),
                                                                                                       lu(EMapTypeConst<T>(icovariance, n, n)),
                                                                                                       gen(std::random_device{}())
{
  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor (default mean = 0, covariance=I)
 * 
 * \param n vector size
 */
template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(int const n) : mean(EVectorX<T>::Zero(n)),
                                                                                 covariance(EMatrixX<T>::Identity(n, n)),
                                                                                 transform(EMatrixX<T>::Identity(n, n)),
                                                                                 lu(EMatrixX<T>::Identity(n, n)),
                                                                                 gen(std::random_device{}()) {}

/*!
 * \brief Move constructor, construct a new multivariateNormalDistribution object from input multivariateNormalDistribution object
 * 
 * \param other  Input multivariateNormalDistribution object
 */
template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(multivariateNormalDistribution<T> &&other) : mean(std::move(other.mean)),
                                                                                                               covariance(std::move(other.covariance)),
                                                                                                               transform(std::move(other.transform)),
                                                                                                               lu(std::move(other.lu)),
                                                                                                               gen(std::move(other.gen)),
                                                                                                               d(std::move(other.d)) {}

/*!
 * \brief Move assignment operator
 * 
 * \param other  Input multivariateNormalDistribution object
 * \return multivariateNormalDistribution& 
 */
template <typename T>
multivariateNormalDistribution<T> &multivariateNormalDistribution<T>::operator=(multivariateNormalDistribution<T> &&other)
{
  mean = std::move(other.mean);
  covariance = std::move(other.covariance);
  transform = std::move(other.transform);
  lu = std::move(other.lu);
  gen = std::move(other.gen);
  d = std::move(other.d);
  return *this;
}

/*!
 * \returns a vector with multivariate normal distribution
 */
template <typename T>
EVectorX<T> multivariateNormalDistribution<T>::operator()()
{
  return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T const x) { return d(gen); });
}

/*!
 * \returns a vector with multivariate normal distribution
 */
template <typename T>
EVectorX<T> multivariateNormalDistribution<T>::dist()
{
  return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T const x) { return d(gen); });
}

} // namespace randomdist
} // namespace umuq

#endif // UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION
