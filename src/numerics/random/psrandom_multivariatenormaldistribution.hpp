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
 * \tparam RealType Floating-point data type 
 * 
 * The [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) of a 
 * \f$n \text{-dimensional}\f$ random vector \f$ {\mathbf X} = \left[X_1, \cdots, X_n \right]^T \f$
 * can be written in the following notation:<br>
 * \f$ {\mathbf X} \sim \mathcal{N} (\mu, \Sigma) \f$<br>
 * with \f$n \text{-dimensional}\f$ mean vector <br>
 * \f$ \mu = {\mathbf E}[{\mathbf X}] = [{\mathbf E}[X_1], \cdots, {\mathbf E}[X_n]]^T, \f$
 * and \f$ n \times n \f$ covariance matrix <br>
 * \f$ \Sigma= {\mathbf E}[({\mathbf X}-\mu)({\mathbf X}-\mu)^T] = [\text{Covariance}[X_i, X_j]; 1\le i,j \le n] \f$ 
 * 
 * Reference:<br>
 * https://en.wikipedia.org/wiki/Multivariate_normal_distribution
 */
template <typename RealType = double>
class multivariateNormalDistribution
{
 public:
   /*!
    * \brief Construct a new multivariateNormalDistribution object
    * 
    * \param Mean        \f$ n \text{-dimensional}\f$ mean vector
    * \param Covariance  \f$ n \times n \f$ covariance matrix
    */
   multivariateNormalDistribution(EVectorX<RealType> const &Mean, EMatrixX<RealType> const &Covariance);

   /*!
    * \brief Construct a new multivariateNormalDistribution object
    * 
    * \param Mean        \f$ n \text{-dimensional}\f$ mean vector
    * \param Covariance  \f$ n \times n \f$ covariance matrix
    * \param n           Vector size (\f$n \text{-dimensional}\f$ vector)
    */
   multivariateNormalDistribution(RealType const *Mean, RealType const *Covariance, int const n);

   /*!
    * \brief Construct a new multivariateNormalDistribution object
    * 
    * \param Covariance  \f$ n \times n \f$ covariance matrix
    */
   explicit multivariateNormalDistribution(EMatrixX<RealType> const &Covariance);

   /*!
    * \brief Construct a new multivariateNormalDistribution object
    * 
    * \param Covariance  \f$ n \times n \f$ covariance matrix
    * \param n           Vector size (\f$n \text{-dimensional}\f$ vector)
    */
   multivariateNormalDistribution(RealType const *Covariance, int const n);

   /*!
    * \brief Construct a new multivariateNormalDistribution object
    * 
    * \param n  Vector size (\f$n \text{-dimensional}\f$ vector)
    */
   explicit multivariateNormalDistribution(int const n);

   /*!
    * \brief Move constructor, construct a new multivariateNormalDistribution object 
    * 
    * \param other multivariateNormalDistribution object
    */
   multivariateNormalDistribution(multivariateNormalDistribution<RealType> &&other);

   /*!
    * \brief Move assignment operator
    * 
    * \param other  multivariateNormalDistribution object
    * 
    * \return multivariateNormalDistribution& 
    */
   multivariateNormalDistribution<RealType> &operator=(multivariateNormalDistribution<RealType> &&other);

   /*!
    * \brief Destroy the multivariateNormalDistribution object
    */
   ~multivariateNormalDistribution();

   /*!
    * \returns A vector of random numbers x according to a multivariate normal distribution \f$ \mathcal{N} (\mu, \Sigma) \f$
    */
   inline EVectorX<RealType> operator()();

   /*!
    * \returns A vector of random numbers x according to a multivariate normal distribution \f$ \mathcal{N} (\mu, \Sigma) \f$
    */
   inline EVectorX<RealType> dist();

   /*!
    * \brief Fill the array of dataPoints with random samples from a multivariate normal distribution \f$ \mathcal{N} (\mu, \Sigma) \f$ 
    * 
    * \param dataPoints          Array of data points, where each point is a \f$n \text{-dimensional}\f$ 
    *                            point (dataPointDimension) and we have nDataPoints of them.
    *                            On return each data point is a random sample according to the multivariate 
    *                            normal distribution \f$ \mathcal{N} (\mu, \Sigma) \f$ 
    * \param dataPointDimension  Data point dimension (\f$n \text{-dimensional}\f$ point)
    * \param nDataPoints         Number of data points
    */
   inline void dist(RealType *dataPoints, int const dataPointDimension, int const nDataPoints);

   /*!
    * \brief Fill the eMatrix with random samples from a multivariate normal distribution \f$ \mathcal{N} (\mu, \Sigma) \f$ 
    * 
    * \param eMatrix  Matrix of random numbers, where each column is an \f$n \text{-dimensional}\f$ point
    *                 and there are number of columns of the eMatrix points. <br>
    *                 On return each column is a random sample according to the multivariate normal 
    *                 distribution \f$ \mathcal{N} (\mu, \Sigma) \f$
    */
   inline void dist(EMatrixX<RealType> &eMatrix);

 private:
   /*!
    * \brief Delete a multivariateNormalDistribution object copy construction
    * 
    * Avoiding implicit generation of the copy constructor.
    */
   multivariateNormalDistribution(multivariateNormalDistribution<RealType> const &) = delete;

   /*!
    * \brief Delete a multivariateNormalDistribution object assignment
    * 
    * Avoiding implicit copy assignment.
    */
   multivariateNormalDistribution<RealType> &operator=(multivariateNormalDistribution<RealType> const &) = delete;

 public:
   /*! Vector of size \f$n\f$ */
   EVectorX<RealType> mean;

   /*! Variance-covariance matrix of size \f$ n \times n \f$ */
   EMatrixX<RealType> covariance;

 private:
   /*! Matrix of size \f$n \times n\f$ */
   EMatrixX<RealType> transform;

 public:
   /*! LU decomposition of a matrix with complete pivoting */
   Eigen::FullPivLU<EMatrixX<RealType>> lu;

 private:
   /*! Generates random numbers according to the Normal (or Gaussian) random number distribution */
   std::normal_distribution<RealType> d;
};

template <typename RealType>
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(EVectorX<RealType> const &Mean, EMatrixX<RealType> const &Covariance) : mean(Mean),
                                                                                                                                                 covariance(Covariance),
                                                                                                                                                 lu(Covariance)
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
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(RealType const *Mean, RealType const *Covariance, int const n) : mean(EMapTypeConst<RealType>(Mean, n, 1)),
                                                                                                                                          covariance(EMapTypeConst<RealType>(Covariance, n, n)),
                                                                                                                                          lu(EMapTypeConst<RealType>(Covariance, n, n))
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
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(EMatrixX<RealType> const &Covariance) : multivariateNormalDistribution(EVectorX<RealType>::Zero(Covariance.rows()), Covariance) {}

template <typename RealType>
multivariateNormalDistribution<RealType>::multivariateNormalDistribution(RealType const *Covariance, int const n) : mean(EVectorX<RealType>::Zero(n)),
                                                                                                                    covariance(EMapTypeConst<RealType>(Covariance, n, n)),
                                                                                                                    lu(EMapTypeConst<RealType>(Covariance, n, n))
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
inline EVectorX<RealType> multivariateNormalDistribution<RealType>::operator()()
{
   int const me = PRNG_initialized ? torc_i_worker_id() : 0;
   return mean + transform * EVectorX<RealType>{mean.size()}.unaryExpr([&](RealType const x) { return d(NumberGenerator[me]); });
}

template <typename RealType>
inline EVectorX<RealType> multivariateNormalDistribution<RealType>::dist()
{
   int const me = PRNG_initialized ? torc_i_worker_id() : 0;
   return mean + transform * EVectorX<RealType>{mean.size()}.unaryExpr([&](RealType const x) { return d(NumberGenerator[me]); });
}

template <typename RealType>
inline void multivariateNormalDistribution<RealType>::dist(RealType *dataPoints, int const dataPointDimension, int const nDataPoints)
{
   int const me = PRNG_initialized ? torc_i_worker_id() : 0;
   EMapType<RealType, Eigen::ColMajor> eMatrix(dataPoints, dataPointDimension, nDataPoints);
   for (auto i = 0; i < nDataPoints; ++i)
   {
      eMatrix.col(i) = mean + transform * EVectorX<RealType>{mean.size()}.unaryExpr([&](RealType const x) { return d(NumberGenerator[me]); });
   }
}

template <typename RealType>
inline void multivariateNormalDistribution<RealType>::dist(EMatrixX<RealType> &eMatrix)
{
   int const me = PRNG_initialized ? torc_i_worker_id() : 0;
   for (auto i = 0; i < eMatrix.cols(); ++i)
   {
      eMatrix.col(i) = mean + transform * EVectorX<RealType>{mean.size()}.unaryExpr([&](RealType const x) { return d(NumberGenerator[me]); });
   }
}

} // namespace randomdist
} // namespace umuq

#endif // UMUQ_PSRANDOM_MULTIVARIATENORMALDISTRIBUTION
