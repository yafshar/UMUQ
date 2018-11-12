#ifndef UMUQ_HYPERCUBESAMPLING_H
#define UMUQ_HYPERCUBESAMPLING_H

#include "../density.hpp"
#include "../random/psrandom.hpp"
#include "../stats.hpp"

namespace umuq
{

/*! \defgroup HypercubeSampling_Module Hypercube sampling module
 * \ingroup Numerics_Module 
 * 
 * This is the Hypercube sampling module of %UMUQ providing all necessary classes for generating a 
 * deterministic, near-random samples or random samples of points over the interior of a hypercube 
 * in N dimensions.
 */

/*! \class hypercubeSampling
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
class hypercubeSampling
{
  public:
	/*!
     * \brief Construct a new hypercubeSampling object
     * 
     * \param TotalNumPoints  Total number of points to generate in the hypercube
     * \param NumDimensions   Number of dimensions
     * \param LowerBound      NumDimensions size vector containing lower bound in each dimension of the hypercube
     * \param UpperBound      NumDimensions size vector containing upper bound in each dimension of the hypercube
     */
	hypercubeSampling(int const TotalNumPoints, int const NumDimensions, double const *LowerBound = nullptr, double const *UpperBound = nullptr);

	/*!
     * \brief Construct a new hypercubeSampling object
     * 
     * \param NumPointsInEachDirection  NumDimensions size vector containing the number of points along each dimension of the hypercube
     * \param NumDimensions             Number of dimensions
     * \param LowerBound                NumDimensions size vector containing lower bound in each dimension of the hypercube
     * \param UpperBound                NumDimensions size vector containing upper bound in each dimension of the hypercube
	 */
	hypercubeSampling(int const *NumPointsInEachDirection, int const NumDimensions, double const *LowerBound = nullptr, double const *UpperBound = nullptr);

	/*!
     * \brief Construct a new hypercubeSampling object
     * 
     * \param NumPointsInEachDirection  NumDimensions size vector containing the number of points along each dimension of the hypercube
     * \param LowerBound                NumDimensions size vector containing lower bound in each dimension of the hypercube
     * \param UpperBound                NumDimensions size vector containing upper bound in each dimension of the hypercube
	 */
	hypercubeSampling(std::vector<int> const &NumPointsInEachDirection,
					  std::vector<double> const &LowerBound = EmptyVector<double>, std::vector<double> const &UpperBound = EmptyVector<double>);

	/*!
	 * \brief Destroy the hypercubeSampling object
	 * 
	 */
	~hypercubeSampling();

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
     * \brief Generates random uniform points in N-dimensional cube of \f$ [lowerBound \cdots upperBound] \f$.
     * 
     * \param points  Uniform random sampling points in the hypercube (It samples totalNumPoints in the hypercube).
     */
	bool uniform(T *&points);

	/*!
     * \brief Generates random uniform points in N-dimensional cube of \f$ [lowerBound \cdots upperBound] \f$.
     * 
     * \param points   Uniform random sampling points in the hypercube (It samples nPoints in the hypercube).
     * \param nPoints  Number of sampling points in the hypercube.
     */
	bool uniform(T *&points, int const nPoints);

	/*!
     * \brief Generates random uniform points in N-dimensional cube of \f$ [lowerBound \cdots upperBound] \f$.
     * 
     * \param points  Uniform random sampling points in the hypercube (It samples totalNumPoints in the hypercube).
     */
	bool uniform(std::vector<T> &points);

	/*!
     * \brief Generates random uniform points in N-dimensional cube of \f$ [lowerBound \cdots upperBound] \f$.
     * 
     * \param points   Uniform random sampling points in the hypercube (It samples nPoints in the hypercube).
     * \param nPoints  Number of sampling points in the hypercube.
     */
	bool uniform(std::vector<T> &points, int const nPoints);

	/*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
     * 
     * \return false If it encounters an unexpected problem
     */
	bool setRandomGenerator(psrandom<double> *PRNG);

  private:
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

  private:
	//! Flat (Uniform) distribution
	std::unique_ptr<uniformDistribution<double>> uniformDist;
};

template <typename T>
hypercubeSampling<T>::hypercubeSampling(int const TotalNumPoints, int const NumDimensions, double const *LowerBound, double const *UpperBound) : totalNumPoints(TotalNumPoints),
																																				 numDimensions(NumDimensions),
																																				 uniformDist(nullptr)
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
hypercubeSampling<T>::hypercubeSampling(int const *NumPointsInEachDirection, int const NumDimensions, double const *LowerBound, double const *UpperBound) : totalNumPoints(std::accumulate(NumPointsInEachDirection, NumPointsInEachDirection + NumDimensions, 1, std::multiplies<int>())),
																																							numPointsInEachDirection(NumPointsInEachDirection, NumPointsInEachDirection + NumDimensions),
																																							numDimensions(NumDimensions),
																																							uniformDist(nullptr)

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
hypercubeSampling<T>::hypercubeSampling(std::vector<int> const &NumPointsInEachDirection, std::vector<double> const &LowerBound, std::vector<double> const &UpperBound) : totalNumPoints(std::accumulate(NumPointsInEachDirection.begin(), NumPointsInEachDirection.end(), 1, std::multiplies<int>())),
																																										  numPointsInEachDirection(NumPointsInEachDirection),
																																										  numDimensions(NumPointsInEachDirection.size()),
																																										  uniformDist(nullptr)

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
hypercubeSampling<T>::~hypercubeSampling() {}

template <typename T>
bool hypercubeSampling<T>::grid(T *&gridPoints, int const Edges)
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
bool hypercubeSampling<T>::grid(std::vector<T> &gridPoints, int const Edges)
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
bool hypercubeSampling<T>::grid(T *&gridPoints, int const *NumPointsInEachDirection, int const Edges)
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
bool hypercubeSampling<T>::grid(std::vector<T> &gridPoints, std::vector<int> const &NumPointsInEachDirection, int const Edges)
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
bool hypercubeSampling<T>::gridInUnitCube(T *&gridPoints, int const Edges)
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
bool hypercubeSampling<T>::gridInUnitCube(std::vector<T> &gridPoints, int const Edges)
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
bool hypercubeSampling<T>::gridInUnitCube(T *&gridPoints, int const *NumPointsInEachDirection, int const Edges)
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
bool hypercubeSampling<T>::gridInUnitCube(std::vector<T> &gridPoints, std::vector<int> const &NumPointsInEachDirection, int const Edges)
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
bool hypercubeSampling<T>::setRandomGenerator(psrandom<double> *PRNG)
{
	if (PRNG)
	{
		if (PRNG_initialized)
		{
			if (isItUnitCube)
			{
				std::vector<double> Lb(numDimensions, 0.);
				std::vector<double> Ub(numDimensions, 1.);
				try
				{
					uniformDist.reset(new uniformDistribution<double>(Lb.data(), Ub.data(), numDimensions * 2));
				}
				catch (...)
				{
					UMUQFAILRETURN("Failed to allocate memory!");
				}
			}
			else
			{
				try
				{
					uniformDist.reset(new uniformDistribution<double>(lowerBound.data(), upperBound.data(), numDimensions * 2));
				}
				catch (...)
				{
					UMUQFAILRETURN("Failed to allocate memory!");
				}
			}
			return uniformDist->setRandomGenerator(PRNG);
		}
		UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to any prior distribution!");
	}
	UMUQFAILRETURN("The pseudo-random number generator is not assigned!");
}

template <typename T>
bool hypercubeSampling<T>::uniform(T *&points)
{
	if (totalNumPoints < 1)
	{
		UMUQFAILRETURN("Wrong number of points of ", totalNumPoints, " < 1 !");
	}

	if (!uniformDist)
	{
		UMUQFAILRETURN("Distribution is not set!");
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
		uniformDist->sample(coord);
		coord += numDimensions;
	}
	return true;
}

template <typename T>
bool hypercubeSampling<T>::uniform(T *&points, int const nPoints)
{
	if (nPoints < 1)
	{
		UMUQFAILRETURN("Wrong number of points of ", nPoints, " < 1 !");
	}

	if (!uniformDist)
	{
		UMUQFAILRETURN("Distribution is not set!");
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
		uniformDist->sample(coord);
		coord += numDimensions;
	}
	return true;
}

template <typename T>
bool hypercubeSampling<T>::uniform(std::vector<T> &points)
{
	if (totalNumPoints < 1)
	{
		UMUQFAILRETURN("Wrong number of points of ", totalNumPoints, " < 1 !");
	}

	if (!uniformDist)
	{
		UMUQFAILRETURN("Distribution is not set!");
	}

	if (points.size() < totalNumPoints * numDimensions)
	{
		points.resize(totalNumPoints * numDimensions);
	}

	T *coord = points.data();
	for (int i = 0; i < totalNumPoints; i++)
	{
		uniformDist->sample(coord);
		coord += numDimensions;
	}
	return true;
}

template <typename T>
bool hypercubeSampling<T>::uniform(std::vector<T> &points, int const nPoints)
{
	if (nPoints < 1)
	{
		UMUQFAILRETURN("Wrong number of points of ", nPoints, " < 1 !");
	}

	if (!uniformDist)
	{
		UMUQFAILRETURN("Distribution is not set!");
	}

	if (points.size() < nPoints * numDimensions)
	{
		points.resize(nPoints * numDimensions);
	}

	T *coord = points.data();
	for (int i = 0; i < nPoints; i++)
	{
		uniformDist->sample(coord);
		coord += numDimensions;
	}
	return true;
}

} // namespace umuq

#endif // UMUQ_HYPERCUBESAMPLING
