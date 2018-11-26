#ifndef UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION_H
#define UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION_H

namespace umuq
{

namespace randomdist
{

/*! \class multivariateNormalDistribution
 * \ingroup Random_Module
 * 
 * \brief Multivariate normal distribution
 * 
 * NOTE: 
 * - This should be called after setting the State of psrandom object \sa psrandom
 * 
 */
template <typename RealType = double>
class multivariateNormalDistribution
{
 public:
   /*!
   * \brief constructor
   *
   * \param inMean        Mean vector of size \f$n\f$
   * \param inCovariance  Input Variance-covariance matrix of size \f$n \times n\f$
   */
   multivariateNormalDistribution(EVectorX<RealType> const &inMean, EMatrixX<RealType> const &inCovariance);

   /*!
   * \brief constructor
   * 
   * \param inMean        Input mean vector of size \f$n\f$
   * \param inCovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n             Vector size
   */
   multivariateNormalDistribution(RealType const *inMean, RealType const *inCovariance, int const n);

   /*!
   * \brief constructor (default mean = 0)
   *
   * \param inCovariance  Input variance-covariance matrix of size \f$n \times n\f$
   */
   explicit multivariateNormalDistribution(EMatrixX<RealType> const &inCovariance);

   /*!
   * \brief constructor (default mean = 0)
   * 
   * \param inCovariance  Input variance-covariance matrix of size \f$n \times n\f$
   * \param n             Vector size
   */
   multivariateNormalDistribution(RealType const *inCovariance, int const n);

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
   multivariateNormalDistribution(multivariateNormalDistribution<RealType> &&other);

   /*!
   * \brief Move assignment operator
   * 
   * \param other  Input multivariateNormalDistribution object
   * \return multivariateNormalDistribution& 
   */
   multivariateNormalDistribution<RealType> &operator=(multivariateNormalDistribution<RealType> &&other);

   /*!
   * \brief Destroy the multivariatenormal Distribution object
   * 
   */
   ~multivariateNormalDistribution();

   /*!
   * \returns a vector with multivariate normal distribution
   */
   EVectorX<RealType> operator()();

   /*!
   * \returns a vector with multivariate normal distribution
   */
   EVectorX<RealType> dist();

 private:
   // Make it noncopyable
   multivariateNormalDistribution(multivariateNormalDistribution<RealType> const &) = delete;

   // Make it not assignable
   multivariateNormalDistribution<RealType> &operator=(multivariateNormalDistribution<RealType> const &) = delete;

 public:
   //! Vector of size \f$n\f$
   EVectorX<RealType> mean;

   //! Variance-covariance matrix of size \f$ n \times n \f$
   EMatrixX<RealType> covariance;

 private:
   //! Matrix of size \f$n \times n\f$
   EMatrixX<RealType> transform;

 public:
   //! LU decomposition of a matrix with complete pivoting
   Eigen::FullPivLU<EMatrixX<RealType>> lu;

 private:
   //! Generates random numbers according to the Normal (or Gaussian) random number distribution
   std::normal_distribution<RealType> d;
};

template <typename RealType>
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(EVectorX<RealType> const &inMean, EMatrixX<RealType> const &inCovariance) : mean(inMean),
                                                                                                                                                     covariance(inCovariance),
                                                                                                                                                     lu(inCovariance)
{
   if (!std::is_floating_point<RealType>::value)
   {
      UMUQFAIL("This type is not supported in this class!");
   }
   // Computes eigenvalues and eigenvectors of selfadjoint matrices.
   Eigen::SelfAdjointEigenSolver<EMatrixX<RealType>> es(covariance);
   transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

template <typename RealType>
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(RealType const *inMean, RealType const *inCovariance, int const n) : mean(EMapTypeConst<RealType>(inMean, n, 1)),
                                                                                                                                              covariance(EMapTypeConst<RealType>(inCovariance, n, n)),
                                                                                                                                              lu(EMapTypeConst<RealType>(inCovariance, n, n))
{
   if (!std::is_floating_point<RealType>::value)
   {
      UMUQFAIL("This type is not supported in this class!");
   }
   // Computes eigenvalues and eigenvectors of selfadjoint matrices.
   Eigen::SelfAdjointEigenSolver<EMatrixX<RealType>> es(covariance);
   transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

template <typename RealType>
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(EMatrixX<RealType> const &inCovariance) : multivariateNormalDistribution(EVectorX<RealType>::Zero(inCovariance.rows()), inCovariance)
{
   if (!std::is_floating_point<RealType>::value)
   {
      UMUQFAIL("This type is not supported in this class!");
   }
}

template <typename RealType>
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(RealType const *inCovariance, int const n) : mean(EVectorX<RealType>::Zero(n)),
                                                                                                                      covariance(EMapTypeConst<RealType>(inCovariance, n, n)),
                                                                                                                      lu(EMapTypeConst<RealType>(inCovariance, n, n))
{
   if (!std::is_floating_point<RealType>::value)
   {
      UMUQFAIL("This type is not supported in this class!");
   }
   // Computes eigenvalues and eigenvectors of selfadjoint matrices.
   Eigen::SelfAdjointEigenSolver<EMatrixX<RealType>> es(covariance);
   transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

template <typename RealType>
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(int const n) : mean(EVectorX<RealType>::Zero(n)),
                                                                                        covariance(EMatrixX<RealType>::Identity(n, n)),
                                                                                        transform(EMatrixX<RealType>::Identity(n, n)),
                                                                                        lu(EMatrixX<RealType>::Identity(n, n))
{
   if (!std::is_floating_point<RealType>::value)
   {
      UMUQFAIL("This type is not supported in this class!");
   }
}

template <typename RealType>
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(multivariateNormalDistribution<RealType> &&other) : mean(std::move(other.mean)),
                                                                                                                             covariance(std::move(other.covariance)),
                                                                                                                             transform(std::move(other.transform)),
                                                                                                                             lu(std::move(other.lu)),
                                                                                                                             d(std::move(other.d)) {}

template <typename RealType>
multivariateNormalDistribution<RealType> &multivariateNormalDistribution<RealType>::operator=(multivariateNormalDistribution<RealType> &&other)
{
   mean = std::move(other.mean);
   covariance = std::move(other.covariance);
   transform = std::move(other.transform);
   lu = std::move(other.lu);
   d = std::move(other.d);
   return *this;
}

template <typename RealType>
multivariateNormalDistribution<RealType>::~multivariateNormalDistribution() {}

template <typename RealType>
EVectorX<RealType> multivariateNormalDistribution<RealType>::operator()()
{
   int const me = PRNG_initialized ? torc_i_worker_id() : 0;
   return mean + transform * EVectorX<RealType>{mean.size()}.unaryExpr([&](RealType const x) { return d(NumberGenerator[me]); });
}

template <typename RealType>
EVectorX<RealType> multivariateNormalDistribution<RealType>::dist()
{
   int const me = PRNG_initialized ? torc_i_worker_id() : 0;
   return mean + transform * EVectorX<RealType>{mean.size()}.unaryExpr([&](RealType const x) { return d(NumberGenerator[me]); });
}

} // namespace randomdist
} // namespace umuq

#endif // UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION
