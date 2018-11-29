#ifndef UMUQ_PSRANDOM_MULTINOMIALDISTRIBUTION_H
#define UMUQ_PSRANDOM_MULTINOMIALDISTRIBUTION_H

namespace umuq
{

namespace randomdist
{

/*! \class multinomialDistribution
 * \ingroup Random_Module
 * 
 * Let \f$ X=\left( X_1, \cdots, X_K \right) \f$ have a multinomial distribution \f$ M_K\left(N, p\right) \f$ <br>
 * The distribution of \f$ X \f$ is given by: <br>
 * \f$
 *     Pr(X_1=n_1, \cdots, X_K=n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \cdots p_K^{n_K}
 * \f$ <br> 
 *
 * where \f$ n_1, \cdots n_K \f$ are nonnegative integers satisfying \f$ sum_{i=1}^{K} {n_i} = N\f$, <br>
 * and \f$p = \left(p_1, \cdots, p_K\right)\f$ is a probability distribution. 
 *
 * Random variates are generated using the conditional binomial method. <br>
 * This scales well with N and does not require a setup step.
 *   
 * Reference: <br>
 * C.S. David, The computer generation of multinomial random variates, <br>
 * Comp. Stat. Data Anal. 16 (1993) 205-217
 */
template <typename RealType = double>
class multinomialDistribution
{
  public:
	/*!
     * \brief Construct a new multinomialDistribution object with a multinomial distribution \f$ M_K\left(N, p\right) \f$
     *
     * \param p  Vector of probabilities \f$ p_1, \cdots, p_k \f$
     * \param K  Size of vector which shows K possible mutually exclusive outcomes 
     * \param N  N independent trials
     */
	multinomialDistribution(RealType const *p, int const K, int const N);

	/*!
     * \brief Move constructor, construct a new multinomialDistribution object from input multinomialDistribution object
     * 
     * \param other  multinomialDistribution object
     */
	multinomialDistribution(multinomialDistribution<RealType> &&other);

	/*!
     * \brief Move assignment operator
     * 
     * \param other  multinomialDistribution object
	 * 
     * \return multinomialDistribution& 
     */
	multinomialDistribution<RealType> &operator=(multinomialDistribution<RealType> &&other);

	/*!
	 * \brief Destroy the multinomial Distribution object
	 * 
	 */
	~multinomialDistribution();

	/*!
     * \returns Vector of random sample from the multinomial distribution \f$ M_K\left(N, p\right) \f$
     */
	EVectorX<int> operator()();

	/*!
     * \returns Vector of random sample from the multinomial distribution \f$ M_K\left(N, p\right) \f$
     */
	EVectorX<int> dist();

	/*!
     * \brief Vector of random sample from the multinomial distribution \f$ M_K\left(N, p\right) \f$
     * 
 	 * \param mndist  Vector of random sample from the multinomial distribution \f$ M_K\left(N, p\right) \f$
     */
	void dist(int *mndist);

	/*!
     * \brief Vector of random sample from the multinomial distribution \f$ M_K\left(N, p\right) \f$
     * 
 	 * \param mndist  Vector of random sample from the multinomial distribution \f$ M_K\left(N, p\right) \f$
     */
	void dist(std::vector<int> &mndist);

	/*!
    * \brief Fill the array of dataPoints with random samples from the multinomial distribution \f$ M_K\left(N, p\right) \f$
    * 
    * \param dataPoints          Array of data points, where each point is a \f$K \text{-dimensional}\f$ 
	*                            point (dataPointDimension) and we have nDataPoints of them.
	*                            On return each data point is a random sample according to the multinomial 
	*                            distribution \f$ M_K\left(N, p\right) \f$
    * \param dataPointDimension  Data point dimension (\f$K \text{-dimensional}\f$ point)
    * \param nDataPoints         Number of data points
    */
	void dist(int *dataPoints, int const dataPointDimension, int const nDataPoints);

	/*!
     * \brief Fill the eMatrix with random samples from the multinomial distribution \f$ M_K\left(N, p\right) \f$
     * 
     * \param eMatrix  Matrix of random numbers, where each column is an \f$K \text{-dimensional}\f$ point
	 *                 and there are number of columns of the eMatrix points. <br>
	 *                 On return each column is a random sample according to the multinomial
	 *                 distribution \f$ M_K\left(N, p\right) \f$
     */
	void dist(EMatrixX<int> &eMatrix);

  private:
	// Make it noncopyable
	multinomialDistribution(multinomialDistribution<RealType> const &) = delete;

	// Make it not assignable
	multinomialDistribution<RealType> &operator=(multinomialDistribution<RealType> const &) = delete;

  private:
	/*! Vector of probabilities \f$ p_1, \cdots, p_k \f$ */
	std::vector<RealType> probabilities;

	/*! Total sum of the probabilities. */
	RealType totalProbabilitySum;

	/*! Number of possible mutually exclusive outcomes */
	int nMutuallyExclusiveOutcomes;

	/*! Number of independent trials*/
	int nIndependentTrials;
};

template <typename RealType>
multinomialDistribution<RealType>::multinomialDistribution(RealType const *p, int const K, int const N) : probabilities(p, p + K),
																										  totalProbabilitySum(std::accumulate(p, p + K, RealType{})),
																										  nMutuallyExclusiveOutcomes(K),
																										  nIndependentTrials(N)
{
	if (!std::is_floating_point<RealType>::value)
	{
		UMUQFAIL("This type is not supported in this class!");
	}
}

template <typename RealType>
multinomialDistribution<RealType>::multinomialDistribution(multinomialDistribution<RealType> &&other) {}

template <typename RealType>
multinomialDistribution<RealType> &multinomialDistribution<RealType>::operator=(multinomialDistribution<RealType> &&other) { return *this; }

template <typename RealType>
multinomialDistribution<RealType>::~multinomialDistribution() {}

template <typename RealType>
EVectorX<int> multinomialDistribution<RealType>::operator()()
{
	int const me = PRNG_initialized ? torc_i_worker_id() : 0;
	EVectorX<int> mndist(nMutuallyExclusiveOutcomes);
	RealType probabilitySum(0);
	int nProbabilitySum(0);
	for (int i = 0; i < nMutuallyExclusiveOutcomes; i++)
	{
		if (probabilities[i] > 0.0)
		{
			std::binomial_distribution<> d(nIndependentTrials - nProbabilitySum, probabilities[i] / (totalProbabilitySum - probabilitySum));
			mndist[i] = d(NumberGenerator[me]);
		}
		else
		{
			mndist[i] = 0;
		}
		probabilitySum += probabilities[i];
		nProbabilitySum += mndist[i];
	}
	return mndist;
}

template <typename RealType>
EVectorX<int> multinomialDistribution<RealType>::dist()
{
	// Get the thread ID
	int const me = PRNG_initialized ? torc_i_worker_id() : 0;
	EVectorX<int> mndist(nMutuallyExclusiveOutcomes);
	RealType probabilitySum(0);
	int nProbabilitySum(0);
	for (int i = 0; i < nMutuallyExclusiveOutcomes; i++)
	{
		if (probabilities[i] > 0.0)
		{
			std::binomial_distribution<> d(nIndependentTrials - nProbabilitySum, probabilities[i] / (totalProbabilitySum - probabilitySum));
			mndist[i] = d(NumberGenerator[me]);
		}
		else
		{
			mndist[i] = 0;
		}
		probabilitySum += probabilities[i];
		nProbabilitySum += mndist[i];
	}
	return mndist;
}

template <typename RealType>
void multinomialDistribution<RealType>::dist(int *mndist)
{
	// Get the thread ID
	int const me = PRNG_initialized ? torc_i_worker_id() : 0;
	RealType probabilitySum(0);
	int nProbabilitySum(0);
	for (int i = 0; i < nMutuallyExclusiveOutcomes; i++)
	{
		if (probabilities[i] > 0.0)
		{
			std::binomial_distribution<> d(nIndependentTrials - nProbabilitySum, probabilities[i] / (totalProbabilitySum - probabilitySum));
			mndist[i] = d(NumberGenerator[me]);
		}
		else
		{
			mndist[i] = 0;
		}
		probabilitySum += probabilities[i];
		nProbabilitySum += mndist[i];
	}
}

template <typename RealType>
void multinomialDistribution<RealType>::dist(std::vector<int> &mndist)
{
#ifdef DEBUG
	if (mndist.size() != static_cast<std::size_t>(nMutuallyExclusiveOutcomes))
	{
		UMUQFAIL("The input array size of ", mndist.size(), " != with the number of possible mutually exclusive outcomes ", nMutuallyExclusiveOutcomes, "!");
	}
#endif
	// Get the thread ID
	int const me = PRNG_initialized ? torc_i_worker_id() : 0;
	RealType probabilitySum(0);
	int nProbabilitySum(0);
	for (int i = 0; i < nMutuallyExclusiveOutcomes; i++)
	{
		if (probabilities[i] > 0.0)
		{
			std::binomial_distribution<> d(nIndependentTrials - nProbabilitySum, probabilities[i] / (totalProbabilitySum - probabilitySum));
			mndist[i] = d(NumberGenerator[me]);
		}
		else
		{
			mndist[i] = 0;
		}
		probabilitySum += probabilities[i];
		nProbabilitySum += mndist[i];
	}
}

template <typename RealType>
void multinomialDistribution<RealType>::dist(int *dataPoints, int const dataPointDimension, int const nDataPoints)
{
#ifdef DEBUG
	if (dataPointDimension != nMutuallyExclusiveOutcomes)
	{
		UMUQFAIL("The input data number of rows of ", dataPointDimension, " != with the number of possible mutually exclusive outcomes ", nMutuallyExclusiveOutcomes, "!");
	}
#endif
	int const me = PRNG_initialized ? torc_i_worker_id() : 0;
	for (auto j = 0, l = 0; j < nDataPoints; ++j)
	{
		RealType probabilitySum(0);
		int nProbabilitySum(0);
		for (int i = 0; i < dataPointDimension; ++i, ++l)
		{
			if (probabilities[i] > 0.0)
			{
				std::binomial_distribution<> d(nIndependentTrials - nProbabilitySum, probabilities[i] / (totalProbabilitySum - probabilitySum));
				dataPoints[l] = d(NumberGenerator[me]);
			}
			else
			{
				dataPoints[l] = 0;
			}
			probabilitySum += probabilities[i];
			nProbabilitySum += dataPoints[l];
		}
	}
}

template <typename RealType>
void multinomialDistribution<RealType>::dist(EMatrixX<int> &eMatrix)
{
#ifdef DEBUG
	if (eMatrix.rows() != static_cast<std::size_t>(nMutuallyExclusiveOutcomes))
	{
		UMUQFAIL("The input matrix number of rows of ", eMatrix.rows(), " != with the number of possible mutually exclusive outcomes ", nMutuallyExclusiveOutcomes, "!");
	}
#endif
	int const me = PRNG_initialized ? torc_i_worker_id() : 0;
	for (auto j = 0; j < eMatrix.cols(); ++j)
	{
		RealType probabilitySum(0);
		int nProbabilitySum(0);
		for (int i = 0; i < nMutuallyExclusiveOutcomes; i++)
		{
			if (probabilities[i] > 0.0)
			{
				std::binomial_distribution<> d(nIndependentTrials - nProbabilitySum, probabilities[i] / (totalProbabilitySum - probabilitySum));
				eMatrix(i, j) = d(NumberGenerator[me]);
			}
			else
			{
				eMatrix(i, j) = 0;
			}
			probabilitySum += probabilities[i];
			nProbabilitySum += eMatrix(i, j);
		}
	}
}

} // namespace randomdist
} // namespace umuq

#endif // UMUQ_PSRANDOM_MULTINOMIALDISTRIBUTION
