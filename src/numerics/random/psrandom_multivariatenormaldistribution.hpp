#ifndef UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION_H
#define UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION_H

namespace umuq
{

namespace randomdist
{

/*! \class multivariatenormalDistribution
 * \ingroup Random_Module
 * 
 * \brief Multivariate normal distribution
 * 
 * \note 
 * - This should be called after setting the State of psrandom object. \sa umuq::random::psrandom.
 * 
 */
template <typename T = double>
class multivariatenormalDistribution
{
public:
  /*!
   * \brief constructor
   *
   * \param inMean        Mean vector of size \f$n\f$
   * \param inCovariance  Input Variance-covariance matrix of size \f$n \times n\f$
   */
  multivariatenormalDistribution(EVectorX<T> const &inMean, EMatrixX<T> const &inCovariance);

  /*!
   * \brief constructor
   * 
   * \param inMean        Input mean vector of size \f$n\f$
   * \param inCovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n             Vector size
   */
  multivariatenormalDistribution(T const *inMean, T const *inCovariance, int const n);

  /*!
   * \brief constructor (default mean = 0)
   *
   * \param inCovariance  Input variance-covariance matrix of size \f$n \times n\f$
   */
  explicit multivariatenormalDistribution(EMatrixX<T> const &inCovariance);

  /*!
   * \brief constructor (default mean = 0)
   * 
   * \param inCovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n             Vector size
   */
  multivariatenormalDistribution(T const *inCovariance, int const n);

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
   * \returns multivariatenormalDistribution& 
   */
  multivariatenormalDistribution<T> &operator=(multivariatenormalDistribution<T> &&other);

  /*!
   * \brief Destroy the multivariatenormal Distribution object
   * 
   */
  ~multivariatenormalDistribution();

  /*!
   * \returns a vector with multivariate normal distribution
   */
  EVectorX<T> operator()();

  /*!
   * \returns a vector with multivariate normal distribution
   */
  EVectorX<T> dist();

private:
  /*!
   * \brief Delete a multivariatenormalDistribution object copy construction
   * 
   * Make it noncopyable.
   */
  multivariatenormalDistribution(multivariatenormalDistribution<T> const &) = delete;

  /*!
   * \brief Delete a multivariatenormalDistribution object assignment
   * 
   * Make it nonassignable
   * 
   * \returns multivariatenormalDistribution<T>& 
   */
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
 * \note 
 * - This can be called without setting the State of psrandom object
 * 
 */
template <typename T = double>
class multivariateNormalDistribution
{
public:
  /*!
   * \brief constructor
   *
   * \param inMean        Mean vector of size \f$n\f$
   * \param inCovariance  Input Variance-covariance matrix of size \f$n \times n\f$
   */
  multivariateNormalDistribution(EVectorX<T> const &inMean, EMatrixX<T> const &inCovariance);

  /*!
   * \brief constructor
   * 
   * \param inMean        Input mean vector of size \f$n\f$
   * \param inCovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n             Vector size
   */
  multivariateNormalDistribution(T const *inMean, T const *inCovariance, int const n);

  /*!
   * \brief constructor (default mean = 0)
   *
   * \param inCovariance  Input variance-covariance matrix of size \f$n \times n\f$
   */
  explicit multivariateNormalDistribution(EMatrixX<T> const &inCovariance);

  /*!
   * \brief constructor (default mean = 0)
   * 
   * \param inCovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n            Vector size
   */
  multivariateNormalDistribution(T const *inCovariance, int const n);

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
   * \returns multivariateNormalDistribution& 
   */
  multivariateNormalDistribution<T> &operator=(multivariateNormalDistribution<T> &&other);

  /*!
   * \brief Destroy the multivariate Normal Distribution object
   * 
   */
  ~multivariateNormalDistribution();

  /*!
   * \returns a vector with multivariate normal distribution
   */
  EVectorX<T> operator()();

  /*!
   * \returns a vector with multivariate normal distribution
   */
  EVectorX<T> dist();

private:

  /*!
   * \brief Delete a multivariateNormalDistribution object copy construction
   * 
   * Make it noncopyable.
   */
  multivariateNormalDistribution(multivariateNormalDistribution<T> const &) = delete;

  /*!
   * \brief Delete a multivariateNormalDistribution object assignment
   * 
   * Make it nonassignable
   * 
   * \returns multivariateNormalDistribution<T>& 
   */
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

template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(EVectorX<T> const &inMean, EMatrixX<T> const &inCovariance) : mean(inMean),
                                                                                                                                covariance(inCovariance),
                                                                                                                                lu(inCovariance)
{
  if (!PRNG_initialized)
  {
    UMUQFAIL("One should set the current state of the engine before constructing this object!");
  }

  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(T const *inMean, T const *inCovariance, int const n) : mean(EMapTypeConst<T>(inMean, n, 1)),
                                                                                                                         covariance(EMapTypeConst<T>(inCovariance, n, n)),
                                                                                                                         lu(EMapTypeConst<T>(inCovariance, n, n))
{
  if (!PRNG_initialized)
  {
    UMUQFAIL("One should set the current state of the engine before constructing this object!");
  }

  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(EMatrixX<T> const &inCovariance) : multivariatenormalDistribution(EVectorX<T>::Zero(inCovariance.rows()), inCovariance)
{
  if (!PRNG_initialized)
  {
    UMUQFAIL("One should set the current state of the engine before constructing this object!");
  }
}

template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(T const *inCovariance, int const n) : mean(EVectorX<T>::Zero(n)),
                                                                                                        covariance(EMapTypeConst<T>(inCovariance, n, n)),
                                                                                                        lu(EMapTypeConst<T>(inCovariance, n, n))
{
  if (!PRNG_initialized)
  {
    UMUQFAIL("One should set the current state of the engine before constructing this object!");
  }

  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

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

template <typename T>
multivariatenormalDistribution<T>::multivariatenormalDistribution(multivariatenormalDistribution<T> &&other) : mean(std::move(other.mean)),
                                                                                                               covariance(std::move(other.covariance)),
                                                                                                               transform(std::move(other.transform)),
                                                                                                               lu(std::move(other.lu)),
                                                                                                               d(std::move(other.d)) {}

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

template <typename T>
multivariatenormalDistribution<T>::~multivariatenormalDistribution() {}

template <typename T>
EVectorX<T> multivariatenormalDistribution<T>::operator()()
{
  int const me = torc_i_worker_id();
  return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T const x) { return d(NumberGenerator[me]); });
}

template <typename T>
EVectorX<T> multivariatenormalDistribution<T>::dist()
{
  int const me = torc_i_worker_id();
  return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T const x) { return d(NumberGenerator[me]); });
}

template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(EVectorX<T> const &inMean, EMatrixX<T> const &inCovariance) : mean(inMean),
                                                                                                                                covariance(inCovariance),
                                                                                                                                lu(inCovariance),
                                                                                                                                gen(std::random_device{}())
{
  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(T const *inMean, T const *inCovariance, int const n) : mean(EMapTypeConst<T>(inMean, n, 1)),
                                                                                                                         covariance(EMapTypeConst<T>(inCovariance, n, n)),
                                                                                                                         lu(EMapTypeConst<T>(inCovariance, n, n)),
                                                                                                                         gen(std::random_device{}())
{
  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(EMatrixX<T> const &inCovariance) : multivariateNormalDistribution(EVectorX<T>::Zero(inCovariance.rows()), inCovariance) {}

template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(T const *inCovariance, int const n) : mean(EVectorX<T>::Zero(n)),
                                                                                                        covariance(EMapTypeConst<T>(inCovariance, n, n)),
                                                                                                        lu(EMapTypeConst<T>(inCovariance, n, n)),
                                                                                                        gen(std::random_device{}())
{
  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
  transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(int const n) : mean(EVectorX<T>::Zero(n)),
                                                                                 covariance(EMatrixX<T>::Identity(n, n)),
                                                                                 transform(EMatrixX<T>::Identity(n, n)),
                                                                                 lu(EMatrixX<T>::Identity(n, n)),
                                                                                 gen(std::random_device{}()) {}

template <typename T>
multivariateNormalDistribution<T>::multivariateNormalDistribution(multivariateNormalDistribution<T> &&other) : mean(std::move(other.mean)),
                                                                                                               covariance(std::move(other.covariance)),
                                                                                                               transform(std::move(other.transform)),
                                                                                                               lu(std::move(other.lu)),
                                                                                                               gen(std::move(other.gen)),
                                                                                                               d(std::move(other.d)) {}

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

template <typename T>
multivariateNormalDistribution<T>::~multivariateNormalDistribution() {}

template <typename T>
EVectorX<T> multivariateNormalDistribution<T>::operator()()
{
  return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T const x) { return d(gen); });
}

template <typename T>
EVectorX<T> multivariateNormalDistribution<T>::dist()
{
  return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T const x) { return d(gen); });
}

} // namespace randomdist
} // namespace umuq

#endif // UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION
