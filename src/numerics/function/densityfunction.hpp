#ifndef UMUQ_DENSITYFUNCTION_H
#define UMUQ_DENSITYFUNCTION_H

#include "../factorial.hpp"
#include "../eigenlib.hpp"
#include "functiontype.hpp"
#include "umuqfunction.hpp"
#include "../random/psrandom.hpp"

namespace umuq
{
/*! \defgroup Density_Module density module
 * \ingroup Numerics_Module
 * 
 * This is the density module of %UMUQ providing all necessary classes for probability density computation.
 */

/*! \namespace umuq::density
 * \ingroup Density_Module
 * 
 * \brief Namespace containing all the functions for probability density computation.
 * 
 */
inline namespace density
{

/*! \class densityFunction
 * \ingroup Density_Module
 * 
 * \brief Density function class
 * 
 * Density function or a probability density (PDF), is a function, with a value at any given point (or sample point) 
 * interpreted as a relative likelihood that the value of the random variable would be equal to that sample.
 * 
 * \tparam T   Data type
 * \tparam F   Function type
 */
template <typename T, class F>
class densityFunction : public umuqFunction<T, F>
{
public:
  /*!
   * \brief Construct a new density Function object
   * 
   * \param Name Distribution name
   */
  explicit densityFunction(char const *Name = "");

  /*!
   * \brief Construct a new density Function object
   * 
   * \param Params     Parameters of density Function object
   * \param NumParams  Number of parameters
   * \param Name       Distribution name
   */
  densityFunction(T const *Params, int const NumParams, char const *Name = "");

  /*!
   * \brief Construct a new density Function object
   * 
   * \param Params     Parameters of density Function object
   * \param NumParams  Number of parameters
   * \param Name       Distribution name
   */
  densityFunction(std::vector<T> const &Params, char const *Name = "");

  /*!
   * \brief Construct a new density Function object
   * 
   * \param Params1     Parameters of density Function object
   * \param Params2     Parameters of density Function object
   * \param NumParams   Number of parameters
   * \param Name        Distribution name
   */
  densityFunction(T const *Params1, T const *Params2, int const NumParams, char const *Name = "");

  /*!
   * \brief Construct a new density Function object
   * 
   * \param Params1     Parameters of density Function object
   * \param Params2     Parameters of density Function object
   * \param NumParams   Number of parameters
   * \param Name        Distribution name
   */
  densityFunction(std::vector<T> const &Params1, std::vector<T> const &Params2, char const *Name = "");

  /*!
   * \brief Move constructor, construct a new density Function object
   * 
   * \param other densityFunction object
   */
  densityFunction(densityFunction<T, F> &&other);

  /*!
   * \brief Move assignment operator
   * 
   * \param other densityFunction object 
   */
  densityFunction<T, F> &operator=(densityFunction<T, F> &&other);

  /*!
   * \brief Destroy the density Function object
   * 
   */
  ~densityFunction();

public:
  /*!
   * \brief Log of density function
   * 
   * \returns the function value (Log of density function)
   */
  F lf;

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Vector of random samples 
   * 
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(T *x);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Vector of random samples 
   *  
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(std::vector<T> &x);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Vector of random samples 
   *  
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(EVectorX<T> &x);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x         Vector of random samples 
   * \param nSamples  Number of sample vectors
   *
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(T *x, int const nSamples);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Vector of random samples 
   * \param nSamples  Number of sample vectors
   * 
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(std::vector<T> &x, int const nSamples);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Matrix of random samples 
   * 
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(EMatrixX<T> &x);

  /*!
   * \brief Set the Random Number Generator object 
   * 
   * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual inline bool setRandomGenerator(psrandom<T> *PRNG);

protected:
  //! Pointer to pseudo random number generator object
  psrandom<T> *prng;
};

template <typename T, class F>
densityFunction<T, F>::densityFunction(char const *Name) : umuqFunction<T, F>(Name), prng(nullptr) {}

template <typename T, class F>
densityFunction<T, F>::densityFunction(T const *Params, int const NumParams, const char *Name) : umuqFunction<T, F>(Params, NumParams, Name), prng(nullptr) {}

template <typename T, class F>
densityFunction<T, F>::densityFunction(std::vector<T> const &Params, const char *Name) : umuqFunction<T, F>(Params, Name), prng(nullptr) {}

template <typename T, class F>
densityFunction<T, F>::densityFunction(T const *Params1, T const *Params2, int const NumParams, const char *Name) : umuqFunction<T, F>(Params1, Params2, NumParams, Name), prng(nullptr) {}

template <typename T, class F>
densityFunction<T, F>::densityFunction(std::vector<T> const &Params1, std::vector<T> const &Params2, const char *Name) : umuqFunction<T, F>(Params1, Params2, Name), prng(nullptr) {}

template <typename T, class F>
densityFunction<T, F>::~densityFunction() {}

template <typename T, class F>
densityFunction<T, F>::densityFunction(densityFunction<T, F> &&other) : umuqFunction<T, F>::umuqFunction(std::move(other)),
                                                                        lf(std::move(other.lf)),
                                                                        prng(other.prng)
{
}

template <typename T, class F>
densityFunction<T, F> &densityFunction<T, F>::operator=(densityFunction<T, F> &&other)
{
  umuqFunction<T, F>::operator=(std::move(other));
  lf = std::move(other.lf);
  prng = other.prng;

  return *this;
}

template <typename T, class F>
bool densityFunction<T, F>::sample(T *x)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename T, class F>
bool densityFunction<T, F>::sample(std::vector<T> &x)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename T, class F>
bool densityFunction<T, F>::sample(EVectorX<T> &x)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename T, class F>
bool densityFunction<T, F>::sample(T *x, int const nSamples)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename T, class F>
bool densityFunction<T, F>::sample(std::vector<T> &x, int const nSamples)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename T, class F>
bool densityFunction<T, F>::sample(EMatrixX<T> &x)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename T, class F>
inline bool densityFunction<T, F>::setRandomGenerator(psrandom<T> *PRNG)
{
  UMUQFAILRETURN("Not implemented!");
}

} // namespace density
} // namespace umuq

#endif // UMUQ_DENSITYFUNCTION_H
