#ifndef UMUQ_DENSITYFUNCTION_H
#define UMUQ_DENSITYFUNCTION_H

#include "../factorial.hpp"
#include "../eigenlib.hpp"
#include "functiontype.hpp"
#include "umuqfunction.hpp"
#include "../random/psrandom.hpp"

namespace umuq
{
/*! 
 * \defgroup Density_Module Density module
 * \ingroup Numerics_Module
 * 
 * This is the density module of %UMUQ providing all necessary classes for probability density computation.
 */

/*! 
 * \namespace umuq::density
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
 * \tparam DataType     Data type
 * \tparam FunctionType Function type
 */
template <typename DataType, class FunctionType>
class densityFunction : public umuqFunction<DataType, FunctionType>
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
  densityFunction(DataType const *Params, int const NumParams, char const *Name = "");

  /*!
   * \brief Construct a new density Function object
   * 
   * \param Params     Parameters of density Function object
   * \param NumParams  Number of parameters
   * \param Name       Distribution name
   */
  densityFunction(std::vector<DataType> const &Params, char const *Name = "");

  /*!
   * \brief Construct a new density Function object
   * 
   * \param Params1     Parameters of density Function object
   * \param Params2     Parameters of density Function object
   * \param NumParams   Number of parameters
   * \param Name        Distribution name
   */
  densityFunction(DataType const *Params1, DataType const *Params2, int const NumParams, char const *Name = "");

  /*!
   * \brief Construct a new density Function object
   * 
   * \param Params1     Parameters of density Function object
   * \param Params2     Parameters of density Function object
   * \param NumParams   Number of parameters
   * \param Name        Distribution name
   */
  densityFunction(std::vector<DataType> const &Params1, std::vector<DataType> const &Params2, char const *Name = "");

  /*!
   * \brief Move constructor, construct a new density Function object
   * 
   * \param other densityFunction object
   */
  densityFunction(densityFunction<DataType, FunctionType> &&other);

  /*!
   * \brief Move assignment operator
   * 
   * \param other densityFunction object 
   */
  densityFunction<DataType, FunctionType> &operator=(densityFunction<DataType, FunctionType> &&other);

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
  FunctionType lf;

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Vector of random samples 
   * 
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(DataType *x);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Vector of random samples 
   *  
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(std::vector<DataType> &x);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Vector of random samples 
   *  
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(EVectorX<DataType> &x);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x         Vector of random samples 
   * \param nSamples  Number of sample vectors
   *
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(DataType *x, int const nSamples);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Vector of random samples 
   * \param nSamples  Number of sample vectors
   * 
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(std::vector<DataType> &x, int const nSamples);

  /*!
   * \brief Create random samples based on the distribution
   * 
   * \param x Matrix of random samples 
   * 
   * \return false If Random Number Generator object is not assigned
   */
  virtual bool sample(EMatrixX<DataType> &x);

  /*!
   * \brief Set the Random Number Generator object 
   * 
   * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual inline bool setRandomGenerator(psrandom<DataType> *PRNG);

  /*!
   * \brief Get the Random Number Generator object 
   * 
   * \returns Pseudo-random number object. \sa umuq::random::psrandom.
   */
  virtual inline psrandom<DataType> *getRandomGenerator();

protected:
  //! Pointer to pseudo random number generator object
  psrandom<DataType> *prng;
};

template <typename DataType, class FunctionType>
densityFunction<DataType, FunctionType>::densityFunction(char const *Name) : umuqFunction<DataType, FunctionType>(Name), prng(nullptr) {}

template <typename DataType, class FunctionType>
densityFunction<DataType, FunctionType>::densityFunction(DataType const *Params, int const NumParams, const char *Name) : umuqFunction<DataType, FunctionType>(Params, NumParams, Name), prng(nullptr) {}

template <typename DataType, class FunctionType>
densityFunction<DataType, FunctionType>::densityFunction(std::vector<DataType> const &Params, const char *Name) : umuqFunction<DataType, FunctionType>(Params, Name), prng(nullptr) {}

template <typename DataType, class FunctionType>
densityFunction<DataType, FunctionType>::densityFunction(DataType const *Params1, DataType const *Params2, int const NumParams, const char *Name) : umuqFunction<DataType, FunctionType>(Params1, Params2, NumParams, Name), prng(nullptr) {}

template <typename DataType, class FunctionType>
densityFunction<DataType, FunctionType>::densityFunction(std::vector<DataType> const &Params1, std::vector<DataType> const &Params2, const char *Name) : umuqFunction<DataType, FunctionType>(Params1, Params2, Name), prng(nullptr) {}

template <typename DataType, class FunctionType>
densityFunction<DataType, FunctionType>::~densityFunction() {}

template <typename DataType, class FunctionType>
densityFunction<DataType, FunctionType>::densityFunction(densityFunction<DataType, FunctionType> &&other) : umuqFunction<DataType, FunctionType>::umuqFunction(std::move(other)),
                                                                                                            lf(std::move(other.lf)),
                                                                                                            prng(other.prng)
{
}

template <typename DataType, class FunctionType>
densityFunction<DataType, FunctionType> &densityFunction<DataType, FunctionType>::operator=(densityFunction<DataType, FunctionType> &&other)
{
  umuqFunction<DataType, FunctionType>::operator=(std::move(other));
  lf = std::move(other.lf);
  prng = other.prng;

  return *this;
}

template <typename DataType, class FunctionType>
bool densityFunction<DataType, FunctionType>::sample(DataType *x)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename DataType, class FunctionType>
bool densityFunction<DataType, FunctionType>::sample(std::vector<DataType> &x)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename DataType, class FunctionType>
bool densityFunction<DataType, FunctionType>::sample(EVectorX<DataType> &x)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename DataType, class FunctionType>
bool densityFunction<DataType, FunctionType>::sample(DataType *x, int const nSamples)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename DataType, class FunctionType>
bool densityFunction<DataType, FunctionType>::sample(std::vector<DataType> &x, int const nSamples)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename DataType, class FunctionType>
bool densityFunction<DataType, FunctionType>::sample(EMatrixX<DataType> &x)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename DataType, class FunctionType>
inline bool densityFunction<DataType, FunctionType>::setRandomGenerator(psrandom<DataType> *PRNG)
{
  UMUQFAILRETURN("Not implemented!");
}

template <typename DataType, class FunctionType>
inline psrandom<DataType> *densityFunction<DataType, FunctionType>::getRandomGenerator()
{
  UMUQFAILRETURNNULL("Not implemented!");
}

} // namespace density
} // namespace umuq

#endif // UMUQ_DENSITYFUNCTION_H
