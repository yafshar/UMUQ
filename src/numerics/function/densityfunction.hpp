#ifndef UMUQ_DENSITYFUNCTION_H
#define UMUQ_DENSITYFUNCTION_H

#include "../factorial.hpp"
#include "../eigenlib.hpp"
#include "umuqfunction.hpp"
#include "../random/psrandom.hpp"

/*! \class densityFunction
 * \brief Density function class
 * 
 * A density function or a probability density (PDF), is a function, with a value at any given point (or sample point) 
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
     */
	densityFunction();

	/*!
     * \brief Construct a new density Function object
     * 
     * \param Params     Parameters of density Function object
     * \param NumParams  Number of parameters
     * \param Name       Distribution name
     */
	densityFunction(T const *Params, int const NumParams, const char *Name = "");

	/*!
     * \brief Destroy the density Function object
     * 
     */
	~densityFunction() {}

	/*!
	 * \brief Move constructor, Construct a new densityFunction object
	 * 
	 * \param other densityFunction object
	 */
	densityFunction(densityFunction<T, F> &&other);

	/*!
	 * \brief Move assignment operator
	 * 
	 */
	densityFunction<T, F> &operator=(densityFunction<T, F> &&other);

  public:
	/*!
     * \brief Log of density function
     * 
     * \returns  Log of density function value 
     */
	F lf;
};

/*!
 * \brief Construct a new density Function object
 * 
 */
template <typename T, class F>
densityFunction<T, F>::densityFunction() {}

template <typename T, class F>
densityFunction<T, F>::densityFunction(densityFunction<T, F> &&other) : umuqFunction<T, F>::umuqFunction(std::move(other)),
																		lf(std::move(other.lf))
{
}

template <typename T, class F>
densityFunction<T, F> &densityFunction<T, F>::operator=(densityFunction<T, F> &&other)
{
	umuqFunction<T, F>::operator=(std::move(other));
	this->lf = std::move(other.lf);

	return *this;
}

/*!
 * \brief Construct a new density Function object
 * 
 * \param Params     Parameters of density Function object
 * \param NumParams  Number of parameters
 * \param Name       Distribution name
 */
template <typename T, class F>
densityFunction<T, F>::densityFunction(T const *Params, int const NumParams, const char *Name) : umuqFunction<T, F>(Params, NumParams, Name) {}

#endif // UMUQ_DENSITYFUNCTION_H
