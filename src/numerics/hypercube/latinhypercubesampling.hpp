#ifndef UMUQ_LATINHYPERCUBESAMPLING_H
#define UMUQ_LATINHYPERCUBESAMPLING_H

#include "core/core.hpp"
#include "numerics/inference/prior/priordistribution.hpp"
#include "numerics/stats.hpp"

#include <cstddef>

#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <utility>

namespace umuq
{

/*! \class latinHypercubeSampling
 * \ingroup HypercubeSampling_Module
 *
 * \brief This class generates sample points over the interior of a hypercube in N dimensions
 *
 * The N-dimensional hypercube is the set of M points \f$ {\mathbf X}_1, {\mathbf X}_2, \cdots, {\mathbf X}_M \f$ such that: <br>
 * \f$ {\mathbf X}_{ij} \in [lowerBound(j), upperBound(j)], \f$ for each dimension j, where the \f$ lowerBound(j) \f$ is the
 * lower bound and \f$ upperBound(j) \f$ is the upperbound of the j-th coordinate.
 *
 * \tparam T Data type
 */
template <typename T>
class latinHypercubeSampling
{
  public:
	/*!
     * \brief Construct a new latinHypercubeSampling object. <br>
	 * By default, when no lower and upper bounds are provided, it creates a unit hypercube in \f$ [0,~1]^N \f$.
	 *
     * \param TotalNumPoints  Total number of points to generate in the hypercube
     * \param NumDimensions   Number of dimensions
     * \param LowerBound      \c NumDimensions size vector containing lower bound in each dimension of the hypercube (default is 0)
     * \param UpperBound      \c NumDimensions size vector containing upper bound in each dimension of the hypercube (default is 1)
     */
	latinHypercubeSampling(int const TotalNumPoints, int const NumDimensions, double const *LowerBound = nullptr, double const *UpperBound = nullptr);

	/*!
     * \brief Construct a new latinHypercubeSampling object. <br>
	 * By default, when no lower and upper bounds are provided, it creates a unit hypercube in \f$ [0,~1]^N \f$.
	 *
     * \param NumPointsInEachDirection  NumDimensions size vector containing the number of points along each dimension of the hypercube
     * \param NumDimensions             Number of dimensions
     * \param LowerBound                \c NumDimensions size vector containing lower bound in each dimension of the hypercube (default is 0)
     * \param UpperBound                \c NumDimensions size vector containing upper bound in each dimension of the hypercube (default is 1)
	 */
	latinHypercubeSampling(int const *NumPointsInEachDirection, int const NumDimensions, double const *LowerBound = nullptr, double const *UpperBound = nullptr);

	/*!
     * \brief Construct a new latinHypercubeSampling object. <br>
	 * By default, when no lower and upper bounds are provided, it creates a unit hypercube in \f$ [0,~1]^N \f$.
	 *
     * \param NumPointsInEachDirection  \c NumDimensions size vector containing the number of points along each dimension of the hypercube
     * \param LowerBound                \c NumDimensions size vector containing lower bound in each dimension of the hypercube (default is 0)
     * \param UpperBound                \c NumDimensions size vector containing upper bound in each dimension of the hypercube (default is 1)
	 */
	latinHypercubeSampling(std::vector<int> const &NumPointsInEachDirection,
					  std::vector<double> const &LowerBound = EmptyVector<double>, std::vector<double> const &UpperBound = EmptyVector<double>);

    /*!
     * \brief Move constructor, construct a new latinHypercubeSampling object
     *
     * \param other latinHypercubeSampling object
     */
    explicit latinHypercubeSampling(latinHypercubeSampling<T> &&other);

    /*!
     * \brief Move assignment operator
     *
     * \param other latinHypercubeSampling object
     *
     * \returns latinHypercubeSampling<T>& latinHypercubeSampling object
     */
    latinHypercubeSampling<T> &operator=(latinHypercubeSampling<T> &&other);

	/*!
	 * \brief Destroy the latinHypercubeSampling object
	 *
	 */
	~latinHypercubeSampling();

	/*!
     * \brief Generates a full factorial sampling plan in N dimensions of the cube between \f$ [lowerBound \cdots upperBound] \f$.
	 * Create conventional grid in a hypercube.
     *
     * \param gridPoints   Full factorial sampling plan
     * \param Edges        If \c Edges=1 the points will be equally spaced from edge to edge (default),
     *                     otherwise they will be in the centres of
     *                     \f$ N = numPointsInEachDirection(1) \times numPointsInEachDirection(2) \times \cdots \times numPointsInEachDirection(numDimensions) \f$
     *                     bins filling the cube.
     */
	bool grid(T *&gridPoints, int const Edges = 1);

	/*!
     * \brief Generates a full factorial sampling plan in N dimensions of the cube between \f$ [lowerBound \cdots upperBound] \f$.
	 * Create conventional grid in a hypercube.
     *
     * \param gridPoints   Full factorial sampling plan
     * \param Edges        If \c Edges=1 the points will be equally spaced from edge to edge (default),
     *                     otherwise they will be in the centres of
     *                     \f$ N = numPointsInEachDirection(1) \times numPointsInEachDirection(2) \times \cdots \times numPointsInEachDirection(numDimensions) \f$
     *                     bins filling the cube.
     */
	bool grid(std::vector<T> &gridPoints, int const Edges = 1);

	/*!
     * \brief Generates a full factorial sampling plan in N dimensions of the cube between \f$ [lowerBound \cdots upperBound] \f$.
	 * Create conventional grid in a hypercube.
     *
     * \param gridPoints                Full factorial sampling plan
     * \param NumPointsInEachDirection  NumDimensions size vector containing the number of points along each dimension of the hypercube
     * \param Edges                     If \c Edges=1 the points will be equally spaced from edge to edge (default),
     *                                  otherwise they will be in the centres of
     *                                  \f$ N = numPointsInEachDirection(1) \times numPointsInEachDirection(2) \times \cdots \times numPointsInEachDirection(numDimensions) \f$
     *                                  bins filling the cube.
     */
	bool grid(T *&gridPoints, int const *NumPointsInEachDirection, int const Edges = 1);

	/*!
     * \brief Generates a full factorial sampling plan in N dimensions of the cube between \f$ [lowerBound \cdots upperBound] \f$.
	 * Create conventional grid in a hypercube.
     *
     * \param gridPoints                Full factorial sampling plan
     * \param NumPointsInEachDirection  NumDimensions size vector containing the number of points along each dimension of the hypercube
     * \param Edges                     If \c Edges=1 the points will be equally spaced from edge to edge (default),
     *                                  otherwise they will be in the centres of
     *                                  \f$ N = numPointsInEachDirection(1) \times numPointsInEachDirection(2) \times \cdots \times numPointsInEachDirection(numDimensions) \f$
     *                                  bins filling the cube.
     */
	bool grid(std::vector<T> &gridPoints, std::vector<int> const &NumPointsInEachDirection, int const Edges = 1);

	/*!
     * \brief Generates a full factorial sampling plan in N dimensions of the unit cube of \f$ [0 \cdots 1]^N \f$.
	 * Create conventional grid in unit hypercube.
     *
     * \param gridPoints   Full factorial sampling plan in a unit cube
     * \param Edges        If \c Edges=1 the points will be equally spaced from edge to edge (default),
     *                     otherwise they will be in the centres of
     *                     \f$ N = numPointsInEachDirection(1) \times numPointsInEachDirection(2) \times \cdots \times numPointsInEachDirection(numDimensions) \f$
     *                     bins filling the unit cube.
     */
	bool gridInUnitCube(T *&gridPoints, int const Edges = 1);

	/*!
     * \brief Generates a full factorial sampling plan in N dimensions of the unit cube of \f$ [0 \cdots 1]^N \f$.
	 * Create conventional grid in unit hypercube.
     *
     * \param gridPoints   Full factorial sampling plan in a unit cube
     * \param Edges        If \c Edges=1 the points will be equally spaced from edge to edge (default),
     *                     otherwise they will be in the centres of
     *                     \f$ N = numPointsInEachDirection(1) \times numPointsInEachDirection(2) \times \cdots \times numPointsInEachDirection(numDimensions) \f$
     *                     bins filling the unit cube.
     */
	bool gridInUnitCube(std::vector<T> &gridPoints, int const Edges = 1);

	/*!
     * \brief Generates a full factorial sampling plan in N dimensions of the unit cube of \f$ [0 \cdots 1]^N \f$.
	 * Create conventional grid in unit hypercube.
     *
     * \param gridPoints                Full factorial sampling plan in a unit cube
     * \param NumPointsInEachDirection  NumDimensions size vector containing the number of points along each dimension of the hypercube
     * \param Edges                     If \c Edges=1 the points will be equally spaced from edge to edge (default),
     *                                  otherwise they will be in the centres of
     *                                  \f$ N = numPointsInEachDirection(1) \times numPointsInEachDirection(2) \times \cdots \times numPointsInEachDirection(numDimensions) \f$
     *                                  bins filling the unit cube.
     */
	bool gridInUnitCube(T *&gridPoints, int const *NumPointsInEachDirection, int const Edges = 1);

	/*!
     * \brief Generates a full factorial sampling plan in N dimensions of the unit cube of \f$ [0 \cdots 1]^N \f$.
	 * Create conventional grid in unit hypercube.
     *
     * \param gridPoints                Full factorial sampling plan in a unit cube
     * \param NumPointsInEachDirection  NumDimensions size vector containing the number of points along each dimension of the hypercube
     * \param Edges                     If \c Edges=1 the points will be equally spaced from edge to edge (default),
     *                                  otherwise they will be in the centres of
     *                                  \f$ N = numPointsInEachDirection(1) \times numPointsInEachDirection(2) \times \cdots \times numPointsInEachDirection(numDimensions) \f$
     *                                  bins filling the unit cube.
     */
	bool gridInUnitCube(std::vector<T> &gridPoints, std::vector<int> const &NumPointsInEachDirection, int const Edges = 1);

	/*!
     * \brief Generates points in N-dimensional hypercube of \f$ [lowerBound \cdots upperBound] \f$ according to the requested distribution,
	 * (Default is the uniform distribution).
     *
     * \param points          Points generated in the hypercube (sampled totalNumPoints in the hypercube) according to the distributionTypes.
     * \param Prior           Prior distribution type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite) \sa umuq::priorTypes
     * \param Param1          First parameter for a prior distribution
     * \param Param2          Second parameter for a prior distribution
     * \param compositeprior  Composite priors type
     */
	bool sample(T *&points, priorTypes const PriorType = priorTypes::UNIFORM, T const *Param1 = nullptr, T const *Param2 = nullptr, priorTypes const *compositeprior = nullptr);

	/*!
     * \brief  Generates points in N-dimensional hypercube of \f$ [lowerBound \cdots upperBound] \f$ according to the requested distribution,
	 * (Default is the uniform distribution).
     *
     * \param points          Points generated in the hypercube (sampled nPoints in the hypercube) according to the distributionTypes.
     * \param Prior           Prior distribution type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite) \sa umuq::priorTypes
     * \param nPoints         Number of sampling points in the hypercube.
     * \param Param1          First parameter for a prior distribution
     * \param Param2          Second parameter for a prior distribution
     * \param compositeprior  Composite priors type
     */
	bool sample(T *&points, int const nPoints, priorTypes const PriorType = priorTypes::UNIFORM, T const *Param1 = nullptr, T const *Param2 = nullptr, priorTypes const *compositeprior = nullptr);

	/*!
     * \brief Generates points in N-dimensional hypercube of \f$ [lowerBound \cdots upperBound] \f$ according to the requested distribution,
	 * (Default is the uniform distribution).
     *
     * \param points          Points generated in the hypercube (sampled totalNumPoints in the hypercube) according to the distributionTypes.
     * \param Prior           Prior distribution type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite) \sa umuq::priorTypes
     * \param Param1          First parameter for a prior distribution
     * \param Param2          Second parameter for a prior distribution
     * \param compositeprior  Composite priors type
     */
	bool sample(std::vector<T> &points, priorTypes const PriorType = priorTypes::UNIFORM,
				std::vector<T> const &Param1 = EmptyVector<T>, std::vector<T> const &Param2 = EmptyVector<T>, std::vector<priorTypes> const &compositeprior = EmptyVector<priorTypes>);

	/*!
     * \brief  Generates points in N-dimensional hypercube of \f$ [lowerBound \cdots upperBound] \f$ according to the requested distribution,
	 * (Default is the uniform distribution).
     *
     * \param points          Points generated in the hypercube (sampled nPoints in the hypercube) according to the distributionTypes.
     * \param Prior           Prior distribution type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite) \sa umuq::priorTypes.
     * \param nPoints         Number of sampling points in the hypercube.
     * \param Param1          First parameter for a prior distribution
     * \param Param2          Second parameter for a prior distribution
     * \param compositeprior  Composite priors type
     */
	bool sample(std::vector<T> &points, int const nPoints, priorTypes const PriorType = priorTypes::UNIFORM,
				std::vector<T> const &Param1 = EmptyVector<T>, std::vector<T> const &Param2 = EmptyVector<T>, std::vector<priorTypes> const &compositeprior = EmptyVector<priorTypes>);

	/*!
     * \brief Set the Random Number Generator object
     *
     * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
     *
     * \return false If it encounters an unexpected problem
     */
	bool setRandomGenerator(psrandom<double> *PRNG);

  private:
    /*!
     * \brief Delete a latinHypercubeSampling object default construction
     *
     */
    latinHypercubeSampling() = delete;

    /*!
     * \brief Delete a latinHypercubeSampling object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    latinHypercubeSampling(latinHypercubeSampling<T> const &) = delete;

    /*!
     * \brief Delete a latinHypercubeSampling object assignment
     *
     * Avoiding implicit copy assignment.
     *
     * \returns latinHypercubeSampling<T>&
     */
    latinHypercubeSampling<T> &operator=(latinHypercubeSampling<T> const &) = delete;

  protected:
	//! Total number of points to generate
	std::size_t totalNumPoints;

	//! Number of points to generate in each direction
	std::vector<int> numPointsInEachDirection;

	//! Number of dimensions
	int const numDimensions;

	//! Lower bound of the hypercube
	std::vector<double> lowerBound;

	//! Upper bound of the hypercube
	std::vector<double> upperBound;

	//! Indicator of the unit hypercube or not
	bool isItUnitCube;

  protected:
	//! Prior distribution object
	priorDistribution<T> prior;
};

template <typename T>
latinHypercubeSampling<T>::latinHypercubeSampling(int const TotalNumPoints, int const NumDimensions, double const *LowerBound, double const *UpperBound) : totalNumPoints(TotalNumPoints),
																																				 numDimensions(NumDimensions),
																																				 prior(numDimensions)
{
	if (numDimensions < 1)
	{
		UMUQFAIL("Wrong dimension of ", numDimensions, " < 1 !");
	}

	isItUnitCube = !(LowerBound && UpperBound);

	if (!isItUnitCube)
	{
		lowerBound.resize(numDimensions);
		std::copy(LowerBound, LowerBound + numDimensions, lowerBound.data());
		upperBound.resize(numDimensions);
		std::copy(UpperBound, UpperBound + numDimensions, upperBound.data());

		for (auto lowerIt = lowerBound.begin(), upperIt = upperBound.begin(); lowerIt != lowerBound.end(); lowerIt++, upperIt++)
		{
			if (*lowerIt >= *upperIt)
			{
				UMUQFAIL("Wrong domain size with lowerbound ", *lowerIt, " > upperbound of ", *upperIt, " !");
			}
		}
	}
}

template <typename T>
latinHypercubeSampling<T>::latinHypercubeSampling(int const *NumPointsInEachDirection, int const NumDimensions, double const *LowerBound, double const *UpperBound) : totalNumPoints(std::accumulate(NumPointsInEachDirection, NumPointsInEachDirection + NumDimensions, 1, std::multiplies<int>())),
																																							numPointsInEachDirection(NumPointsInEachDirection, NumPointsInEachDirection + NumDimensions),
																																							numDimensions(NumDimensions),
																																							prior(numDimensions)

{
	{
		umuq::stats s;
		if (s.minelement<int>(NumPointsInEachDirection, NumDimensions) < 2)
		{
			UMUQFAIL("You must have at least two points per dimension!");
		}
	}

	isItUnitCube = !(LowerBound && UpperBound);

	if (!isItUnitCube)
	{
		lowerBound.resize(numDimensions);
		std::copy(LowerBound, LowerBound + numDimensions, lowerBound.data());
		upperBound.resize(numDimensions);
		std::copy(UpperBound, UpperBound + numDimensions, upperBound.data());

		for (auto lowerIt = lowerBound.begin(), upperIt = upperBound.begin(); lowerIt != lowerBound.end(); lowerIt++, upperIt++)
		{
			if (*lowerIt >= *upperIt)
			{
				UMUQFAIL("Wrong domain size with lowerbound ", *lowerIt, " > upperbound of ", *upperIt, " !");
			}
		}
	}
}

template <typename T>
latinHypercubeSampling<T>::latinHypercubeSampling(std::vector<int> const &NumPointsInEachDirection, std::vector<double> const &LowerBound, std::vector<double> const &UpperBound) : totalNumPoints(std::accumulate(NumPointsInEachDirection.begin(), NumPointsInEachDirection.end(), 1, std::multiplies<int>())),
																																										  numPointsInEachDirection(NumPointsInEachDirection),
																																										  numDimensions(NumPointsInEachDirection.size()),
																																										  prior(numDimensions)

{
	{
		umuq::stats s;
		if (s.minelement<int>(numPointsInEachDirection) < 2)
		{
			UMUQFAIL("You must have at least two points per dimension!");
		}
	}

	isItUnitCube = !(LowerBound.size() && UpperBound.size());

	if (!isItUnitCube)
	{
		if (LowerBound.size() != UpperBound.size())
		{
			UMUQFAIL("Wrong vector size !");
		}

		if (LowerBound.size() != numDimensions)
		{
			UMUQFAIL("Wrong vector size !");
		}

		lowerBound.resize(numDimensions);
		std::copy(LowerBound.begin(), LowerBound.end(), lowerBound.begin());
		upperBound.resize(numDimensions);
		std::copy(UpperBound.begin(), UpperBound.end(), upperBound.begin());

		for (auto lowerIt = lowerBound.begin(), upperIt = upperBound.begin(); lowerIt != lowerBound.end(); lowerIt++, upperIt++)
		{
			if (*lowerIt >= *upperIt)
			{
				UMUQFAIL("Wrong domain size with lowerbound ", *lowerIt, " > upperbound of ", *upperIt, " !");
			}
		}
	}
}

template <typename T>
latinHypercubeSampling<T>::latinHypercubeSampling(latinHypercubeSampling<T> &&other)
{
	totalNumPoints = other.totalNumPoints;
	numPointsInEachDirection = std::move(other.numPointsInEachDirection);
	numDimensions = other.numDimensions;
	lowerBound = std::move(other.lowerBound);
	upperBound = std::move(other.upperBound);
	isItUnitCube = other.isItUnitCube;
	prior = std::move(other.prior);
}

template <typename T>
latinHypercubeSampling<T> &latinHypercubeSampling<T>::operator=(latinHypercubeSampling<T> &&other)
{
	totalNumPoints = other.totalNumPoints;
	numPointsInEachDirection = std::move(other.numPointsInEachDirection);
	numDimensions = other.numDimensions;
	lowerBound = std::move(other.lowerBound);
	upperBound = std::move(other.upperBound);
	isItUnitCube = other.isItUnitCube;
	prior = std::move(other.prior);

    return *this;
}

template <typename T>
latinHypercubeSampling<T>::~latinHypercubeSampling() {}

template <typename T>
bool latinHypercubeSampling<T>::grid(T *&gridPoints, int const Edges)
{
	if (isItUnitCube)
	{
		return gridInUnitCube(gridPoints, Edges);
	}

	if (numPointsInEachDirection.size() == 0)
	{
		UMUQFAILRETURN("Uniform grid requires number of points in each direction!");
	}

	if (gridPoints == nullptr)
	{
		try
		{
			gridPoints = new T[totalNumPoints * numDimensions];
		}
		catch (...)
		{
			UMUQFAILRETURN("Failed to allocate memory!");
		}
	}

	// Temporary array
	std::vector<T> Column(totalNumPoints);

	for (int i = 0; i < numDimensions; i++)
	{
		std::vector<double> oneDimensionSlice(numPointsInEachDirection[i]);

		if (Edges == 1)
		{
			double const increment = (upperBound[i] - lowerBound[i]) / (numPointsInEachDirection[i] - 1);
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), lowerBound[i]);
			for (int j = 0; j < numPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}
		else
		{
			double const increment = (upperBound[i] - lowerBound[i]) / numPointsInEachDirection[i];
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), lowerBound[i] + increment / 2.);
			for (int j = 0; j < numPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}

		int const m = std::accumulate(numPointsInEachDirection.data() + i + 1, numPointsInEachDirection.data() + numDimensions, 1, std::multiplies<int>());

		int nPoints = 0;
		while (nPoints < totalNumPoints)
		{
			for (int l = 0; l < numPointsInEachDirection[i]; l++)
			{
				T const fillValue = static_cast<T>(oneDimensionSlice[l]);
				std::fill(Column.data() + nPoints, Column.data() + nPoints + m, fillValue);
				nPoints += m;
			}
		}

		nPoints = 0;

		T *coord = gridPoints + i;
		for (int k = 0; k < totalNumPoints; k++, nPoints += numDimensions)
		{
			coord[nPoints] = Column[k];
		}
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::grid(std::vector<T> &gridPoints, int const Edges)
{
	if (isItUnitCube)
	{
		return gridInUnitCube(gridPoints, Edges);
	}

	if (numPointsInEachDirection.size() == 0)
	{
		UMUQFAILRETURN("Uniform grid requires number of points in each direction!");
	}

	if (gridPoints.size() < totalNumPoints * numDimensions)
	{
		gridPoints.resize(totalNumPoints * numDimensions);
	}

	// Temporary array
	std::vector<T> Column(totalNumPoints);

	for (int i = 0; i < numDimensions; i++)
	{
		std::vector<double> oneDimensionSlice(numPointsInEachDirection[i]);

		if (Edges == 1)
		{
			double const increment = (upperBound[i] - lowerBound[i]) / (numPointsInEachDirection[i] - 1);
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), lowerBound[i]);
			for (int j = 0; j < numPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}
		else
		{
			double const increment = (upperBound[i] - lowerBound[i]) / numPointsInEachDirection[i];
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), lowerBound[i] + increment / 2.);
			for (int j = 0; j < numPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}

		int const m = std::accumulate(numPointsInEachDirection.data() + i + 1, numPointsInEachDirection.data() + numDimensions, 1, std::multiplies<int>());

		int nPoints = 0;
		while (nPoints < totalNumPoints)
		{
			for (int l = 0; l < numPointsInEachDirection[i]; l++)
			{
				T const fillValue = static_cast<T>(oneDimensionSlice[l]);
				std::fill(Column.data() + nPoints, Column.data() + nPoints + m, fillValue);
				nPoints += m;
			}
		}

		nPoints = 0;

		T *coord = gridPoints.data() + i;
		for (int k = 0; k < totalNumPoints; k++, nPoints += numDimensions)
		{
			coord[nPoints] = Column[k];
		}
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::grid(T *&gridPoints, int const *NumPointsInEachDirection, int const Edges)
{
	if (isItUnitCube)
	{
		return gridInUnitCube(gridPoints, NumPointsInEachDirection, Edges);
	}

	if (!NumPointsInEachDirection)
	{
		UMUQFAILRETURN("Pointer is not assigned to a reference!");
	}

	std::size_t const TotalNumPoints = std::accumulate(NumPointsInEachDirection, NumPointsInEachDirection + numDimensions, 1, std::multiplies<int>());

	if (gridPoints == nullptr)
	{
		try
		{
			gridPoints = new T[TotalNumPoints * numDimensions];
		}
		catch (...)
		{
			UMUQFAILRETURN("Failed to allocate memory!");
		}
	}

	// Temporary array
	std::vector<T> Column(TotalNumPoints);

	for (int i = 0; i < numDimensions; i++)
	{
		std::vector<double> oneDimensionSlice(NumPointsInEachDirection[i]);

		if (Edges == 1)
		{
			double const increment = (upperBound[i] - lowerBound[i]) / (NumPointsInEachDirection[i] - 1);
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), lowerBound[i]);
			for (int j = 0; j < NumPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}
		else
		{
			double const increment = (upperBound[i] - lowerBound[i]) / NumPointsInEachDirection[i];
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), lowerBound[i] + increment / 2.);
			for (int j = 0; j < NumPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}

		int const m = std::accumulate(NumPointsInEachDirection + i + 1, NumPointsInEachDirection + numDimensions, 1, std::multiplies<int>());

		int nPoints = 0;
		while (nPoints < TotalNumPoints)
		{
			for (int l = 0; l < NumPointsInEachDirection[i]; l++)
			{
				T const fillValue = static_cast<T>(oneDimensionSlice[l]);
				std::fill(Column.data() + nPoints, Column.data() + nPoints + m, fillValue);
				nPoints += m;
			}
		}

		nPoints = 0;

		T *coord = gridPoints + i;
		for (int k = 0; k < TotalNumPoints; k++, nPoints += numDimensions)
		{
			coord[nPoints] = Column[k];
		}
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::grid(std::vector<T> &gridPoints, std::vector<int> const &NumPointsInEachDirection, int const Edges)
{
	if (isItUnitCube)
	{
		return gridInUnitCube(gridPoints, NumPointsInEachDirection, Edges);
	}

	if (NumPointsInEachDirection.size() == 0)
	{
		UMUQFAILRETURN("Uniform grid requires number of points in each direction!");
	}

	if (NumPointsInEachDirection.size() != numDimensions)
	{
		UMUQFAILRETURN("Wrong size!");
	}

	std::size_t const TotalNumPoints = std::accumulate(NumPointsInEachDirection.begin(), NumPointsInEachDirection.end(), 1, std::multiplies<int>());

	if (gridPoints.size() < TotalNumPoints * numDimensions)
	{
		gridPoints.resize(TotalNumPoints * numDimensions);
	}

	// Temporary array
	std::vector<T> Column(TotalNumPoints);

	for (int i = 0; i < numDimensions; i++)
	{
		std::vector<double> oneDimensionSlice(NumPointsInEachDirection[i]);

		if (Edges == 1)
		{
			double const increment = (upperBound[i] - lowerBound[i]) / (NumPointsInEachDirection[i] - 1);
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), lowerBound[i]);
			for (int j = 0; j < NumPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}
		else
		{
			double const increment = (upperBound[i] - lowerBound[i]) / NumPointsInEachDirection[i];
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), lowerBound[i] + increment / 2.);
			for (int j = 0; j < NumPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}

		int const m = std::accumulate(NumPointsInEachDirection.data() + i + 1, NumPointsInEachDirection.data() + numDimensions, 1, std::multiplies<int>());

		int nPoints = 0;
		while (nPoints < TotalNumPoints)
		{
			for (int l = 0; l < NumPointsInEachDirection[i]; l++)
			{
				T const fillValue = static_cast<T>(oneDimensionSlice[l]);
				std::fill(Column.data() + nPoints, Column.data() + nPoints + m, fillValue);
				nPoints += m;
			}
		}

		nPoints = 0;

		T *coord = gridPoints.data() + i;
		for (int k = 0; k < TotalNumPoints; k++, nPoints += numDimensions)
		{
			coord[nPoints] = Column[k];
		}
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::gridInUnitCube(T *&gridPoints, int const Edges)
{
	if (!isItUnitCube)
	{
		UMUQFAILRETURN("Wrong call! this subroutine is generating uniform mesh in a unit hypercube!");
	}

	if (numPointsInEachDirection.size() == 0)
	{
		UMUQFAILRETURN("Uniform grid requires number of points in each direction!");
	}

	if (gridPoints == nullptr)
	{
		try
		{
			gridPoints = new double[totalNumPoints * numDimensions];
		}
		catch (...)
		{
			UMUQFAILRETURN("Failed to allocate memory!");
		}
	}

	// Temporary array
	std::vector<T> Column(totalNumPoints);

	for (int i = 0; i < numDimensions; i++)
	{
		std::vector<double> oneDimensionSlice(numPointsInEachDirection[i]);

		if (Edges == 1)
		{
			double const increment = 1. / (numPointsInEachDirection[i] - 1);
			for (int j = 0; j < numPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] = j * increment;
			}
		}
		else
		{
			double const increment = 1. / numPointsInEachDirection[i];
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), increment / 2.);
			for (int j = 0; j < numPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}

		int const m = std::accumulate(numPointsInEachDirection.data() + i + 1, numPointsInEachDirection.data() + numDimensions, 1, std::multiplies<int>());

		int nPoints = 0;
		while (nPoints < totalNumPoints)
		{
			for (int l = 0; l < numPointsInEachDirection[i]; l++)
			{
				T const fillValue = static_cast<T>(oneDimensionSlice[l]);
				std::fill(Column.data() + nPoints, Column.data() + nPoints + m, fillValue);
				nPoints += m;
			}
		}

		nPoints = 0;

		T *coord = gridPoints + i;
		for (int k = 0; k < totalNumPoints; k++, nPoints += numDimensions)
		{
			coord[nPoints] = Column[k];
		}
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::gridInUnitCube(std::vector<T> &gridPoints, int const Edges)
{
	if (!isItUnitCube)
	{
		UMUQFAILRETURN("Wrong call! this subroutine is generating uniform mesh in a unit hypercube!");
	}

	if (numPointsInEachDirection.size() == 0)
	{
		UMUQFAILRETURN("Uniform grid requires number of points in each direction!");
	}

	if (gridPoints.size() < totalNumPoints * numDimensions)
	{
		gridPoints.resize(totalNumPoints * numDimensions);
	}

	// Temporary array
	std::vector<T> Column(totalNumPoints);

	for (int i = 0; i < numDimensions; i++)
	{
		std::vector<double> oneDimensionSlice(numPointsInEachDirection[i]);

		if (Edges == 1)
		{
			double const increment = 1.0 / (numPointsInEachDirection[i] - 1);
			for (int j = 0; j < numPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] = j * increment;
			}
		}
		else
		{
			double const increment = 1.0 / numPointsInEachDirection[i];
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), increment / 2.);
			for (int j = 0; j < numPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}

		int const m = std::accumulate(numPointsInEachDirection.data() + i + 1, numPointsInEachDirection.data() + numDimensions, 1, std::multiplies<int>());

		int nPoints = 0;
		while (nPoints < totalNumPoints)
		{
			for (int l = 0; l < numPointsInEachDirection[i]; l++)
			{
				T const fillValue = static_cast<T>(oneDimensionSlice[l]);
				std::fill(Column.data() + nPoints, Column.data() + nPoints + m, fillValue);
				nPoints += m;
			}
		}

		nPoints = 0;

		T *coord = gridPoints.data() + i;
		for (int k = 0; k < totalNumPoints; k++, nPoints += numDimensions)
		{
			coord[nPoints] = Column[k];
		}
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::gridInUnitCube(T *&gridPoints, int const *NumPointsInEachDirection, int const Edges)
{
	if (!isItUnitCube)
	{
		UMUQFAILRETURN("Wrong call! this subroutine is generating uniform mesh in a unit hypercube!");
	}

	if (!NumPointsInEachDirection)
	{
		UMUQFAILRETURN("Pointer is not correctly assigned to a reference!");
	}

	std::size_t const TotalNumPoints = std::accumulate(NumPointsInEachDirection, NumPointsInEachDirection + numDimensions, 1, std::multiplies<int>());

	if (gridPoints == nullptr)
	{
		try
		{
			gridPoints = new double[TotalNumPoints * numDimensions];
		}
		catch (...)
		{
			UMUQFAILRETURN("Failed to allocate memory!");
		}
	}

	// Temporary array
	std::vector<T> Column(TotalNumPoints);

	for (int i = 0; i < numDimensions; i++)
	{
		std::vector<double> oneDimensionSlice(NumPointsInEachDirection[i]);

		if (Edges == 1)
		{
			double const increment = 1. / (NumPointsInEachDirection[i] - 1);
			for (int j = 0; j < NumPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] = j * increment;
			}
		}
		else
		{
			double const increment = 1. / NumPointsInEachDirection[i];
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), increment / 2.);
			for (int j = 0; j < NumPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}

		int const m = std::accumulate(NumPointsInEachDirection + i + 1, NumPointsInEachDirection + numDimensions, 1, std::multiplies<int>());

		int nPoints = 0;
		while (nPoints < TotalNumPoints)
		{
			for (int l = 0; l < NumPointsInEachDirection[i]; l++)
			{
				T const fillValue = static_cast<T>(oneDimensionSlice[l]);
				std::fill(Column.data() + nPoints, Column.data() + nPoints + m, fillValue);
				nPoints += m;
			}
		}

		nPoints = 0;

		T *coord = gridPoints + i;
		for (int k = 0; k < TotalNumPoints; k++, nPoints += numDimensions)
		{
			coord[nPoints] = Column[k];
		}
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::gridInUnitCube(std::vector<T> &gridPoints, std::vector<int> const &NumPointsInEachDirection, int const Edges)
{
	if (!isItUnitCube)
	{
		UMUQFAILRETURN("Wrong call! this subroutine is generating uniform mesh in a unit hypercube!");
	}

	if (NumPointsInEachDirection.size() == 0)
	{
		UMUQFAILRETURN("Uniform grid requires number of points in each direction!");
	}

	if (NumPointsInEachDirection.size() != numDimensions)
	{
		UMUQFAILRETURN("Wrong size!");
	}

	std::size_t const TotalNumPoints = std::accumulate(NumPointsInEachDirection.begin(), NumPointsInEachDirection.end(), 1, std::multiplies<int>());

	if (gridPoints.size() < TotalNumPoints * numDimensions)
	{
		gridPoints.resize(TotalNumPoints * numDimensions);
	}

	// Temporary array
	std::vector<T> Column(TotalNumPoints);

	for (int i = 0; i < numDimensions; i++)
	{
		std::vector<double> oneDimensionSlice(NumPointsInEachDirection[i]);

		if (Edges == 1)
		{
			double const increment = 1.0 / (NumPointsInEachDirection[i] - 1);
			for (int j = 0; j < NumPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] = j * increment;
			}
		}
		else
		{
			double const increment = 1.0 / NumPointsInEachDirection[i];
			std::fill(oneDimensionSlice.begin(), oneDimensionSlice.end(), increment / 2.);
			for (int j = 0; j < NumPointsInEachDirection[i]; j++)
			{
				oneDimensionSlice[j] += j * increment;
			}
		}

		int const m = std::accumulate(NumPointsInEachDirection.data() + i + 1, NumPointsInEachDirection.data() + numDimensions, 1, std::multiplies<int>());

		int nPoints = 0;
		while (nPoints < TotalNumPoints)
		{
			for (int l = 0; l < NumPointsInEachDirection[i]; l++)
			{
				T const fillValue = static_cast<T>(oneDimensionSlice[l]);
				std::fill(Column.data() + nPoints, Column.data() + nPoints + m, fillValue);
				nPoints += m;
			}
		}

		nPoints = 0;

		T *coord = gridPoints.data() + i;
		for (int k = 0; k < TotalNumPoints; k++, nPoints += numDimensions)
		{
			coord[nPoints] = Column[k];
		}
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::setRandomGenerator(psrandom<double> *PRNG)
{
	if (PRNG)
	{
		if (PRNG_initialized)
		{
			if (isItUnitCube)
			{
				std::vector<double> Lb(numDimensions, 0.);
				std::vector<double> Ub(numDimensions, 1.);

				// Set the distribution parameters
				if (!prior.set(Lb, Ub))
				{
					UMUQFAILRETURN("Failed to set the distribution!");
				}
			}
			else
			{
				// Set the distribution parameters
				if (!prior.set(lowerBound, upperBound))
				{
					UMUQFAILRETURN("Failed to set the distribution!");
				}
			}
			return prior.setRandomGenerator(PRNG);
		}
		UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to any prior distribution!");
	}
	UMUQFAILRETURN("The pseudo-random number generator is not assigned!");
}

template <typename T>
bool latinHypercubeSampling<T>::sample(T *&points, priorTypes const PriorType, T const *Param1, T const *Param2, priorTypes const *compositeprior)
{
	if (totalNumPoints < 1)
	{
		UMUQFAILRETURN("Wrong number of points of ", totalNumPoints, " < 1 !");
	}

	if (prior.getpriorType() != PriorType)
	{
		// Pseudo-random number generator
		auto *prng = prior.getRandomGenerator();

		if (!prng)
		{
			UMUQFAILRETURN("The pseudo-random number generator is not assigned!");
		}

		// Construct a prior Distribution object
		prior = std::move(priorDistribution<T>(numDimensions, PriorType));

		// Set the prior parameters
		if (!prior.set(Param1, Param2, compositeprior))
		{
			UMUQFAILRETURN("Failed to set the prior distribution!");
		}

		// Set the Random Number Generator object in the prior
		if (!prior.setRandomGenerator(prng))
		{
			UMUQFAILRETURN("Failed to set the Random Number Generator object in the prior!");
		}
	}

	if (points == nullptr)
	{
		try
		{
			points = new T[totalNumPoints * numDimensions];
		}
		catch (...)
		{
			UMUQFAILRETURN("Failed to allocate memory!");
		}
	}

	T *coord = points;
	for (int i = 0; i < totalNumPoints; i++)
	{
		prior.sample(coord);
		coord += numDimensions;
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::sample(T *&points, int const nPoints, priorTypes const PriorType, T const *Param1, T const *Param2, priorTypes const *compositeprior)
{
	if (nPoints < 1)
	{
		UMUQFAILRETURN("Wrong number of points of ", nPoints, " < 1 !");
	}

	if (prior.getpriorType() != PriorType)
	{
		// Pseudo-random number generator
		auto *prng = prior.getRandomGenerator();

		if (!prng)
		{
			UMUQFAILRETURN("The pseudo-random number generator is not assigned!");
		}

		// Construct a prior Distribution object
		prior = std::move(priorDistribution<T>(numDimensions, PriorType));

		// Set the prior parameters
		if (!prior.set(Param1, Param2, compositeprior))
		{
			UMUQFAILRETURN("Failed to set the prior distribution!");
		}

		// Set the Random Number Generator object in the prior
		if (!prior.setRandomGenerator(prng))
		{
			UMUQFAILRETURN("Failed to set the Random Number Generator object in the prior!");
		}
	}

	if (points == nullptr)
	{
		try
		{
			points = new T[nPoints * numDimensions];
		}
		catch (...)
		{
			UMUQFAILRETURN("Failed to allocate memory!");
		}
	}

	T *coord = points;
	for (int i = 0; i < nPoints; i++)
	{
		prior.sample(coord);
		coord += numDimensions;
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::sample(std::vector<T> &points, priorTypes const PriorType, std::vector<T> const &Param1, std::vector<T> const &Param2, std::vector<priorTypes> const &compositeprior)
{
	if (totalNumPoints < 1)
	{
		UMUQFAILRETURN("Wrong number of points of ", totalNumPoints, " < 1 !");
	}

	if (prior.getpriorType() != PriorType)
	{
		// Pseudo-random number generator
		auto *prng = prior.getRandomGenerator();

		if (!prng)
		{
			UMUQFAILRETURN("The pseudo-random number generator is not assigned!");
		}

		// Construct a prior Distribution object
		prior = std::move(priorDistribution<T>(numDimensions, PriorType));

		// Set the prior parameters
		if (!prior.set(Param1, Param2, compositeprior))
		{
			UMUQFAILRETURN("Failed to set the prior distribution!");
		}

		// Set the Random Number Generator object in the prior
		if (!prior.setRandomGenerator(prng))
		{
			UMUQFAILRETURN("Failed to set the Random Number Generator object in the prior!");
		}
	}

	if (points.size() < totalNumPoints * numDimensions)
	{
		points.resize(totalNumPoints * numDimensions);
	}

	T *coord = points.data();
	for (int i = 0; i < totalNumPoints; i++)
	{
		prior.sample(coord);
		coord += numDimensions;
	}
	return true;
}

template <typename T>
bool latinHypercubeSampling<T>::sample(std::vector<T> &points, int const nPoints, priorTypes const PriorType, std::vector<T> const &Param1, std::vector<T> const &Param2, std::vector<priorTypes> const &compositeprior)
{
	if (nPoints < 1)
	{
		UMUQFAILRETURN("Wrong number of points of ", nPoints, " < 1 !");
	}

	if (prior.getpriorType() != PriorType)
	{
		// Pseudo-random number generator
		auto *prng = prior.getRandomGenerator();

		if (!prng)
		{
			UMUQFAILRETURN("The pseudo-random number generator is not assigned!");
		}

		// Construct a prior Distribution object
		prior = std::move(priorDistribution<T>(numDimensions, PriorType));

		// Set the prior parameters
		if (!prior.set(Param1, Param2, compositeprior))
		{
			UMUQFAILRETURN("Failed to set the prior distribution!");
		}

		// Set the Random Number Generator object in the prior
		if (!prior.setRandomGenerator(prng))
		{
			UMUQFAILRETURN("Failed to set the Random Number Generator object in the prior!");
		}
	}

	if (points.size() < nPoints * numDimensions)
	{
		points.resize(nPoints * numDimensions);
	}

	T *coord = points.data();
	for (int i = 0; i < nPoints; i++)
	{
		prior.sample(coord);
		coord += numDimensions;
	}
	return true;
}

} // namespace umuq

#endif // UMUQ_LATINHYPERCUBESAMPLING_H
