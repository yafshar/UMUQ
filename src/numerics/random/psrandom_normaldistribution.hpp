#ifndef UMUQ_PSRANDOM_NORMALDISTRIBUTION_H
#define UMUQ_PSRANDOM_NORMALDISTRIBUTION_H

namespace umuq
{

/*! \namespace umuq::randomdist
 * \ingroup Random_Module
 * 
 * \brief Namespace containing all necessary classes that generate random number distributions
 */
namespace randomdist
{

/*! \class normalDistribution
 * \ingroup Random_Module
 * 
 * \brief Generates random numbers according to the Normal (or Gaussian) random number distribution
 * 
 * NOTE: 
 * - This should be called after setting the State of psrandom object
 * 
 */
template <typename RealType = double>
class normalDistribution
{
   public:
     /*!
      * \brief Construct a new normalDistribution object (default mean = 0, stddev = 1)
      * 
      * \param mean    Mean
      * \param stddev  Standard deviation
      * 
      */
     normalDistribution(RealType mean = RealType{}, RealType stddev = RealType{1});

     /*!
      * \brief Move constructor, construct a new normalDistribution object from input normalDistribution object
      * 
      * \param other  Input normalDistribution object
      */
     normalDistribution(normalDistribution<RealType> &&other);

     /*!
      * \brief Move assignment operator
      * 
      * \param other  Input normalDistribution object
      * \return normalDistribution& 
      */
     normalDistribution<RealType> &operator=(normalDistribution<RealType> &&other);

     /*!
      * \brief Random numbers x according to Normal (or Gaussian) random number distribution
      * The result type generated by the generator is undefined if RealType is not one of float, 
      * double, or long double
      * 
      * \return Random numbers x according to Normal (or Gaussian) random number distribution
      * 
      */
     inline RealType operator()();

     /*!
      * \brief Random numbers x according to Normal (or Gaussian) random number distribution
      * The result type generated by the generator is undefined if RealType is not one of float, 
      * double, or long double
      * 
      * \return Random numbers x according to Normal (or Gaussian) random number distribution
      * 
      */
     inline RealType dist();

   private:
     // Make it noncopyable
     normalDistribution(normalDistribution<RealType> const &) = delete;

     // Make it not assignable
     normalDistribution<RealType> &operator=(normalDistribution<RealType> const &) = delete;

   private:
     //! Random numbers according to the Normal (or Gaussian) random number distribution
     std::normal_distribution<RealType> d;
};

template <typename RealType>
normalDistribution<RealType>::normalDistribution(RealType mean, RealType stddev) : d(mean, stddev)
{
     if (!std::is_floating_point<RealType>::value)
     {
          UMUQFAIL("This type is not supported in this class!");
     }
}

template <typename RealType>
normalDistribution<RealType>::normalDistribution(normalDistribution<RealType> &&other) : d(std::move(other.d)) {}

template <typename RealType>
normalDistribution<RealType> &normalDistribution<RealType>::operator=(normalDistribution<RealType> &&other)
{
     d = std::move(other.d);
     return *this;
}

template <typename RealType>
inline RealType normalDistribution<RealType>::operator()()
{
     // Get the thread ID
     int const me = PRNG_initialized ? torc_i_worker_id() : 0;
     return d(NumberGenerator[me]);
}

template <typename RealType>
inline RealType normalDistribution<RealType>::dist()
{
     // Get the thread ID
     int const me = PRNG_initialized ? torc_i_worker_id() : 0;
     return d(NumberGenerator[me]);
}

} // namespace randomdist
} // namespace umuq

#endif // UMUQ_PSRANDOM_NORMALDISTRIBUTION
