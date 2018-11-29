#ifndef UMUQ_PSRANDOM_UNIFORMDISTRIBUTION_H
#define UMUQ_PSRANDOM_UNIFORMDISTRIBUTION_H

namespace umuq
{

namespace randomdist
{

/*! \class uniformDistribution
 * \ingroup Random_Module
 * 
 * \brief Produces random floating-point values, uniformly distributed on the interval \f$ [a, b). \f$ 
 */
template <typename RealType = double>
class uniformDistribution
{
   public:
     /*!
      * \brief Construct a new uniformDistribution object (default a = 0, b = 1)
      *
      * \param a  Lower bound of the interval
      * \param b  Upper bound of the interval
      */
     uniformDistribution(RealType const a = RealType{}, RealType const b = RealType{1});

     /*!
      * \brief Move constructor, construct a new uniformDistribution object from input uniformDistribution object
      * 
      * \param other  Input uniformDistribution object
      */
     uniformDistribution(uniformDistribution<RealType> &&other);

     /*!
      * \brief Move assignment operator
      * 
      * \param other  Input uniformDistribution object
      * \return uniformDistribution& 
      */
     uniformDistribution<RealType> &operator=(uniformDistribution<RealType> &&other);

     /*!
      * \brief Destroy the uniform Distribution object
      * 
      */
     ~uniformDistribution();

     /*!
      * \brief Random numbers x according to a uniform distribution
      * The result type generated by the generator is undefined if RealType is not one of float, 
      * double, or long double
      * 
      * \return Random numbers x according to uniform distribution
      */
     inline RealType operator()();

     /*!
      * \brief Random numbers x according to a uniform distribution
      * The result type generated by the generator is undefined if RealType is not one of float, 
      * double, or long double
      * 
      * \return Random numbers x according to a uniform distribution
      */
     inline RealType dist();

     /*!
      * \brief Random numbers x according to a uniform distribution
      * The result type generated by the generator is undefined if RealType is not one of float, 
      * double, or long double
      * 
	 * \param idata  Array of data of random numbers x according to a uniform distribution
	 * \param nSize  Size of the array 
      */
     inline void dist(RealType *idata, int const nSize);

     /*!
      * \brief Random numbers x according to a uniform distribution
      * The result type generated by the generator is undefined if RealType is not one of float, 
      * double, or long double
      * 
	 * \param idata  Array of data of random numbers x according to a uniform distribution
      */
     inline void dist(std::vector<RealType> &idata);

   private:
     // Make it noncopyable
     uniformDistribution(uniformDistribution<RealType> const &) = delete;

     // Make it not assignable
     uniformDistribution<RealType> &operator=(uniformDistribution<RealType> const &) = delete;

   private:
     /*! Random numbers according to the uniform distribution */
     std::uniform_real_distribution<RealType> d;
};

template <typename RealType>
uniformDistribution<RealType>::uniformDistribution(RealType const a, RealType const b) : d(a, b)
{
     if (!std::is_floating_point<RealType>::value)
     {
          UMUQFAIL("This type is not supported in this class!");
     }
}

template <typename RealType>
uniformDistribution<RealType>::uniformDistribution(uniformDistribution<RealType> &&other) : d(std::move(other.d)) {}

template <typename RealType>
uniformDistribution<RealType> &uniformDistribution<RealType>::operator=(uniformDistribution<RealType> &&other)
{
     d = std::move(other.d);
     return *this;
}

template <typename RealType>
uniformDistribution<RealType>::~uniformDistribution(){}

template <typename RealType>
inline RealType uniformDistribution<RealType>::operator()()
{
     // Get the thread ID
     int const me = PRNG_initialized ? torc_i_worker_id() : 0;
     return d(NumberGenerator[me]);
}

template <typename RealType>
inline RealType uniformDistribution<RealType>::dist()
{
     // Get the thread ID
     int const me = PRNG_initialized ? torc_i_worker_id() : 0;
     return d(NumberGenerator[me]);
}

template <typename RealType>
inline void uniformDistribution<RealType>::dist(RealType *idata, int const nSize)
{
     // Get the thread ID
     int const me = PRNG_initialized ? torc_i_worker_id() : 0;
     for (auto i = 0; i < nSize; ++i)
     {
          idata[i] = d(NumberGenerator[me]);
     }
}

template <typename RealType>
inline void uniformDistribution<RealType>::dist(std::vector<RealType> &idata)
{
     // Get the thread ID
     int const me = PRNG_initialized ? torc_i_worker_id() : 0;
     for (auto i = 0; i < idata.size(); ++i)
     {
          idata[i] = d(NumberGenerator[me]);
     }
}

} // namespace randomdist
} // namespace umuq

#endif // UMUQ_PSRANDOM_NORMALDISTRIBUTION
