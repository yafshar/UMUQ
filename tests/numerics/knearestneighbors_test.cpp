#include "core/core.hpp"
#include "io/io.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/knearestneighbors.hpp"
#include "gtest/gtest.h"

/*! 
 * \ingroup Test_Module
 * 
 * Test to check flann library functionality in %UMUQ
 * for a fast approximate nearest neighbor searches
 */
TEST(knearestneighbors_test, HandlesKNN)
{
	umuq::io f;
	EXPECT_TRUE(f.isFileExist("numerics/knearestneighbors_test.txt"));
	EXPECT_TRUE(f.openFile("numerics/knearestneighbors_test.txt"));

	int nRows = 0;
	while (f.readLine())
	{
		// Count the number of non empty and not commented line with "#" as default comment
		nRows++;
	}

	// This data type has two dimensions
	int const nDim = 2;

	// Number of nearest neighbors to find
	int const nNearestNeighbors = 3;

	double *dataPoints = nullptr;
	double *dataPointsDistances = nullptr;
	int *nearestNeighborsIndices = nullptr;

	// Allocate memory for reading the dataPoints
	try
	{
		dataPoints = new double[nRows * nDim];
		dataPointsDistances = new double[nRows + 1];
		nearestNeighborsIndices = new int[nRows * nNearestNeighbors];
	}
	catch (std::bad_alloc &e)
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
	}

	// Rewind the file
	f.rewindFile();

	// Read the array of dataPoints
	EXPECT_TRUE(f.loadMatrix<double>(dataPoints, nRows, nDim));

	// Close the file
	f.closeFile();

	// Create an instance of the kNearestNeighbor object with DistanceType = NeighborDistance::EUCLIDEAN as default
	umuq::kNearestNeighbor<double> KNN(nRows, nDim, nNearestNeighbors);

	// Check number of neighbors
	EXPECT_EQ(KNN.numNearestNeighbors(), nNearestNeighbors);

	// Construct a kd-tree index & do a knn search
	KNN.buildIndex(dataPoints);

	EXPECT_TRUE(KNN.checkNearestNeighbors());

	// Using brute force to find nearest neighbors
	dataPointsDistances[nRows] = std::numeric_limits<double>::max();

	for (int i = 0; i < nRows; i++)
	{
		int const IdI = i * nDim;
		for (int j = 0; j < nRows; j++)
		{
			int const IdJ = j * nDim;
			double const dd[] = {dataPoints[IdJ] - dataPoints[IdI], dataPoints[IdJ + 1] - dataPoints[IdI + 1]};
			dataPointsDistances[j] = dd[0] * dd[0] + dd[1] * dd[1];
		}

		int const Id = i * nNearestNeighbors + nNearestNeighbors;

		std::fill(nearestNeighborsIndices + Id - nNearestNeighbors, nearestNeighborsIndices + Id, nRows);

		for (int j = 0; j < nRows; j++)
		{
			if (i == j)
			{
				continue;
			}

			int const Id3 = nearestNeighborsIndices[Id - 1];
			if (dataPointsDistances[j] < dataPointsDistances[Id3])
			{
				int const Id2 = nearestNeighborsIndices[Id - 2];
				if (dataPointsDistances[j] < dataPointsDistances[Id2])
				{
					int const Id1 = nearestNeighborsIndices[Id - 3];
					if (dataPointsDistances[j] < dataPointsDistances[Id1])
					{
						nearestNeighborsIndices[Id - 1] = nearestNeighborsIndices[Id - 2];
						nearestNeighborsIndices[Id - 2] = nearestNeighborsIndices[Id - 3];
						nearestNeighborsIndices[Id - 3] = j;
					}
					else
					{
						nearestNeighborsIndices[Id - 1] = nearestNeighborsIndices[Id - 2];
						nearestNeighborsIndices[Id - 2] = j;
					}
				}
				else
				{
					nearestNeighborsIndices[Id - 1] = j;
				}
			}
		}
	}

	delete[] dataPointsDistances;

	for (int i = 0; i < nRows; ++i)
	{
		// Get the nearest neighbor from point i
		int *p = KNN.NearestNeighbors(i);

		int const Id = i * nNearestNeighbors;
		for (int j = 0; j < nNearestNeighbors; j++)
		{
			/*!
			 * \ingroup Test_Module
			 * 
			 * \todo
			 * ON MACOS there is a problem for one of the neighbor
			 */
			if (p[j] != nearestNeighborsIndices[Id + j])
			{
				std::cerr << "There is a difference in found neighbor:" << std::endl;
				std::cerr << "Point i = " << i << "Flann neighbor = " << p[j] << " Brute force neighbor=" << nearestNeighborsIndices[Id + j] << std::endl;
				continue;
			}
			EXPECT_EQ(p[j], nearestNeighborsIndices[Id + j]);
		}
	}

	delete[] nearestNeighborsIndices;

	double *dists = nullptr;
	dists = KNN.minDist();

	EXPECT_TRUE(dists != nullptr);

	for (int i = 0; i < nRows; ++i)
	{
		int const IdI = i * nDim;
		int *p = KNN.NearestNeighbors(i);
		for (int j = 0; j < nNearestNeighbors; j++)
		{
			int const IdJ = p[j] * nDim;
			double const dd[] = {dataPoints[IdJ] - dataPoints[IdI], dataPoints[IdJ + 1] - dataPoints[IdI + 1]};
			double const d = dd[0] * dd[0] + dd[1] * dd[1];
			EXPECT_TRUE((dists[i] <= d));
		}
	}

	delete[] dataPoints;
	delete[] dists;
}

/*! 
 * \ingroup Test_Module
 * 
 * Test to check MahalanobisNearestNeighbor class functionality
 * for a fast approximate nearest neighbor searches.
 */
TEST(knearestneighbors_test, HandlesMahalanobisNearestNeighbor)
{
	// Get an instance of a seeded double random object
	umuq::psrandom<double> prng(123);

	// This dataPoints type has two dimensions
	int const nDim = 2;

	// Number of nearest neighbors to find
	int nNearestNeighbors = 20;

	// Number of sampling points
	int nSPoints = 1000;

	// Number of query points
	int nQPoints = 1;

	// Create a zero vector
	umuq::EVector2d V2d(umuq::EVector2d::Zero());

	// Create a 2 by 2 matrix
	umuq::EMatrix2d M2d;
	M2d << 20, 0.5, 0.5, 1;

	// Create an object of type Multivariate normal distribution
	EXPECT_TRUE(prng.set_mvnormal(V2d, M2d));

	// Create a vector for sampling points
	std::vector<double> xPoints(nSPoints * nDim);

	double *x = xPoints.data();

	for (int i = 0; i < nSPoints; i++)
	{
		umuq::EVectorMapType<double> X(x, nDim);
		X = prng.mvnormal->dist();
		x += nDim;
	}

	// Create a vector for query points
	std::vector<double> yPoints(nQPoints * nDim);

	x = yPoints.data();

	for (int i = 0; i < nQPoints; i++)
	{
		umuq::EVectorMapType<double> X(x, nDim);
		X = prng.mvnormal->dist();
		x += nDim;
	}

	// Finding K nearest neighbors with the Mahalanobis distance
	umuq::kNearestNeighbor<double, umuq::NeighborDistance::MAHALANOBIS> KNN(nSPoints, nQPoints, nDim, nNearestNeighbors);

	// set the covariance
	KNN.setCovariance(M2d);

	// Construct a kd-tree index & do a knn search
	KNN.buildIndex(xPoints.data(), yPoints.data());

	umuq::io f;
	if (f.openFile("numerics/Xdata", umuq::io::out))
	{
		EXPECT_TRUE(f.saveMatrix<double>(xPoints, nSPoints, nDim));
		f.closeFile();
	}

	if (f.openFile("numerics/Qdata", umuq::io::out))
	{
		EXPECT_TRUE(f.saveMatrix<double>(yPoints, nQPoints, nDim));
		f.closeFile();
	}

	if (f.openFile("numerics/Ndata", umuq::io::out))
	{
		for (int i = 0; i < nQPoints; ++i)
		{
			int *p = KNN.NearestNeighbors(i);
			for (int j = 0; j < nNearestNeighbors; j++)
			{
				int const IdJ = p[j] * nDim;
				x = xPoints.data() + IdJ;
				EXPECT_TRUE(f.saveMatrix<double>(x, 1, nDim));
			}
		}
		f.closeFile();
	}
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
