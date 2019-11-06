#ifndef UMUQ_PSRANDOM_GAMMADISTRIBUTION_H
#define UMUQ_PSRANDOM_GAMMADISTRIBUTION_H

#include "core/core.hpp"

#include <vector>
#include <random>
#include <type_traits>
#include <utility>

namespace umuq
{

namespace randomdist
{

/*! \class gammaDistribution
 * \ingroup Random_Module
 *
 * \brief Generates random positive values x, distributed according to probability density function \f$ \frac{1}{\Gamma (\alpha) \beta^\alpha}x^{\alpha-1}e^{\frac{-x}{\beta}}.\f$
 * where \f$ \alpha > 0 \f$ is known as the shape parameter and \f$ \beta > 0 \f$ is known as the scale parameter.
 *
 * \tparam RealType Floating-point data type
 *
 * \note
 * - \f$ \alpha > 0 \f$
 * - \f$ \beta > 0 \f$
 */
template <typename RealType = double>
class gammaDistribution
{
   public:
     /*!
      * \brief Construct a new gamma Distribution object
      *
      */
     gammaDistribution();

     /*!
      * \brief Construct a new Gamma distribution object
      *
      * \param alpha  Shape parameter \f$\alpha\f$
      * \param beta   Scale parameter \f$ beta\f$
      */
     gammaDistribution(RealType const alpha, RealType const beta = RealType{1});

     /*!
      * \brief Move constructor, construct a new gammaDistribution object from input gammaDistribution object
      *
      * \param other  Input gammaDistribution object
      */
     gammaDistribution(gammaDistribution<RealType> &&other);

     /*!
      * \brief Move assignment operator
      *
      * \param other  Input gammaDistribution object
      * \return gammaDistribution&
      */
     gammaDistribution<RealType> &operator=(gammaDistribution<RealType> &&other);

     /*!
      * \brief Destroy the gamma Distribution object
      *
      */
     ~gammaDistribution();

     /*!
      * \brief Random numbers x according to probability density function \f$ \lambda e^{-\lambda x} \f$
      * The result type generated by the generator is undefined if RealType is not one of float,
      * double, or long double
      *
      * \return Random numbers x according to probability density function \f$ \lambda e^{-\lambda x} \f$
      */
     inline RealType operator()();

     /*!
      * \brief Random numbers x according to probability density function \f$ \lambda e^{-\lambda x} \f$
      * The result type generated by the generator is undefined if RealType is not one of float,
      * double, or long double
      *
      * \return Random numbers x according to probability density function \f$ \lambda e^{-\lambda x} \f$
      */
     inline RealType dist();

     /*!
      * \brief Random numbers x according to probability density function \f$ \lambda e^{-\lambda x} \f$
      * The result type generated by the generator is undefined if RealType is not one of float,
      * double, or long double
      *
      * \param idata  Array of data of random numbers x according to probability density function \f$ \lambda e^{-\lambda x} \f$
      * \param nSize  Size of the array
      */
     inline void dist(RealType *idata, int const nSize);

     /*!
      * \brief Random numbers x according to probability density function \f$ \lambda e^{-\lambda x} \f$
      * The result type generated by the generator is undefined if RealType is not one of float,
      * double, or long double
      *
      * \param idata  Array of data of random numbers x according to probability density function \f$ \lambda e^{-\lambda x} \f$
      */
     inline void dist(std::vector<RealType> &idata);

   private:
     /*!
      * \brief Delete a gammaDistribution object copy construction
      *
      * Avoiding implicit generation of the copy constructor.
      */
     gammaDistribution(gammaDistribution<RealType> const &) = delete;

     /*!
      * \brief Delete a gammaDistribution object assignment
      *
      * Avoiding implicit copy assignment.
      */
     gammaDistribution<RealType> &operator=(gammaDistribution<RealType> const &) = delete;

   private:
     /*! Random numbers according to to probability density function \f$ \lambda e^{-\lambda x} \f$ */
     std::gamma_distribution<RealType> d;
};

template <typename RealType>
gammaDistribution<RealType>::gammaDistribution() : d(RealType{1}, RealType{1})
{
     if (!std::is_floating_point<RealType>::value)
     {
          UMUQFAIL("This type is not supported in this class!");
     }
}

template <typename RealType>
gammaDistribution<RealType>::gammaDistribution(RealType const alpha, RealType const beta) : d(alpha, beta)
{
     if (!std::is_floating_point<RealType>::value)
     {
          UMUQFAIL("This type is not supported in this class!");
     }
}

template <typename RealType>
gammaDistribution<RealType>::gammaDistribution(gammaDistribution<RealType> &&other) : d(std::move(other.d)) {}

template <typename RealType>
gammaDistribution<RealType> &gammaDistribution<RealType>::operator=(gammaDistribution<RealType> &&other)
{
     d = std::move(other.d);
     return *this;
}

template <typename RealType>
gammaDistribution<RealType>::~gammaDistribution() {}

template <typename RealType>
inline RealType gammaDistribution<RealType>::operator()()
{
     // Get the thread ID
     int const me = PRNG_initialized ? torc_i_worker_id() : 0;
     return d(NumberGenerator[me]);
}

template <typename RealType>
inline RealType gammaDistribution<RealType>::dist()
{
     // Get the thread ID
     int const me = PRNG_initialized ? torc_i_worker_id() : 0;
     return d(NumberGenerator[me]);
}

template <typename RealType>
inline void gammaDistribution<RealType>::dist(RealType *idata, int const nSize)
{
     // Get the thread ID
     int const me = PRNG_initialized ? torc_i_worker_id() : 0;
     for (auto i = 0; i < nSize; ++i)
     {
          idata[i] = d(NumberGenerator[me]);
     }
}

template <typename RealType>
inline void gammaDistribution<RealType>::dist(std::vector<RealType> &idata)
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

#endif // UMUQ_PSRANDOM_GAMMADISTRIBUTION
