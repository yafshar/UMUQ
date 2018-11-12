#ifndef UMUQ_LATINHYPERCUBE_H
#define UMUQ_LATINHYPERCUBE_H

#include <memory>
#include <iostream>
#include <algorithm>
#include <numeric>

#include "../io/io.hpp"
#include "random/psrandom.hpp"
#include "../inference/prior/priordistribution.hpp"

namespace umuq
{

template <typename T>
class latinHypercube
{
  public:
	/*!
     * \brief Construct a new 1 dimensional latinHypercube object
     * 
     */
	latinHypercube();

	/*!
     * \brief Construct a new latinHypercube object
     * 
     * \param nDim       Number of dimensions
     * \param priorType  Prior type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite)
     */
	latinHypercube(int const nDim, int const priorType = 0);

	/*!
     * \brief Create random samples based on the distribution
     * 
     * \param x Vector of random samples 
     * 
     * \return false If Random Number Generator object is not assigned
     */
	bool sample(T *x, int const nSamples);

	/*!
     * \brief Create random samples based on the distribution
     * 
     * \param x Vector of random samples 
     * 
     * \return false If Random Number Generator object is not assigned
     */
	bool sample(std::vector<T> &x, int const nSamples);

	/*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
     * 
     * \return false If it encounters an unexpected problem
     */
	inline bool setRandomGenerator(psrandom<T> *PRNG);

  public:
	//! Prior distribution object
	priorDistribution<T> prior;

  private:
	//! Prior parameter 1
	std::vector<T> priorParam1;

	//! Prior parameter 2
	std::vector<T> priorParam2;
};

template <typename T>
latinHypercube<T>::latinHypercube() : prior(1, 0),
									  priorParam1(1, 0),
									  priorParam2(2, 1)
{
	if (!prior.set(priorParam1, priorParam2))
	{
		UMUQFAIL("Failed to set the default prior parameters");
	}
}

template <typename T>
latinHypercube<T>::latinHypercube(int const nDim, int const priorType) : prior(nDim, priorType),
																		 priorParam1(nDim, 0),
																		 priorParam2(nDim, 1)
{
	if (!prior.set(priorParam1, priorParam2))
	{
		UMUQFAIL("Failed to set the default prior parameters");
	}
}

template <typename T>
inline bool latinHypercube<T>::setRandomGenerator(psrandom<T> *PRNG)
{
	return prior.setRandomGenerator(PRNG);
}

// Generate a latin hypercube sample of size nSamples from this distribution.
template <typename T>
bool latinHypercube<T>::sample(T *x, int const nSamples)
{
#ifdef DEBUG
	if (prior.prng)
	{
#endif
		int const nDim = prior.getDim();

		// Get samples of the random variable
		EMatrixX<T> rawSamples(nDim, nSamples);

		prior.sample(rawSamples);

		// The latin hypercube samples
		EMatrixX<T> latinHypercubeSamples(nDim, nSamples);

		EVectorX<T> samplesMu = rawSamples.rowwise().sum() / nSamples;
		EMatrixX<T> samplesCov = (rawSamples.colwise() - samplesMu) * (rawSamples.colwise() - samplesMu).transpose();
		samplesCov /= nSamples;

		// Sort each dimension of the samples and store the index
		std::vector<unsigned int> Idx(nSamples);

		for (int d = 0; d < nDim; ++d)
		{
			// Set the indices to the identity
			std::iota(Idx.begin(), Idx.end(), 0);

			// Sort the indices based on this dimension of the sample
			std::sort(Idx.begin(), Idx.end(), [&rawSamples, &d](unsigned int i1, unsigned int i2) { return rawSamples(d, i1) < rawSamples(d, i2); });

			EVectorX<T> x(nSamples);
			for (unsigned int i = 0; i < nSamples; ++i)
			{
				x(Idx[i]) = i + 0.5;
			}
			x /= nSamples;

			// Add a random perturbation to the sample and get the inverse RandomGenerator::GetUniform()
			for (unsigned int i = 0; i < nSamples; ++i)
			{
				latinHypercubeSamples(d, i) = InverseCDF(d, x(i));
			}
		}

		return latinHypercubeSamples;
#ifdef DEBUG
	}
	UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
#endif
}

} // namespace umuq

#endif // UMUQ_LATINHYPERCUBE
