#include "core/core.hpp"
#include "numerics/stats.hpp"
#include "gtest/gtest.h"

/*! 
 * \ingroup Test_Module
 * 
 * Test to check stats functionality
 */
TEST(stats_test, HandlesStats)
{
	// Create an instance of stats object
	umuq::stats s;

	// test for normal array of data
	int iArray[] = {2, 3, 5, 7, 1, 6, 8, 10, 9, 4};

	EXPECT_EQ(s.minelement<int>(iArray, 10), 1);
	EXPECT_EQ(s.maxelement<int>(iArray, 10), 10);
	EXPECT_EQ(s.minelement_index<int>(iArray, 10), 4);
	EXPECT_EQ(s.maxelement_index<int>(iArray, 10), 7);

	double sum = s.sum<int, double>(iArray, 10);
	double mean = s.mean<int, double>(iArray, 10);
	double stddev = s.stddev<int, double>(iArray, 10);

	EXPECT_DOUBLE_EQ(sum, 55.0);
	EXPECT_DOUBLE_EQ(mean, 5.5);
	EXPECT_DOUBLE_EQ(stddev, 3.027650354097491);

	// Check for std::vector
	std::vector<int> jArray(20);

	for (int i = 0; i < 20; ++i)
	{
		jArray[i] = iArray[i % 10];
	}

	EXPECT_EQ(s.minelement<int>(jArray), 1);
	EXPECT_EQ(s.maxelement<int>(jArray), 10);
	EXPECT_EQ(s.minelement_index<int>(jArray), 4);
	EXPECT_EQ(s.maxelement_index<int>(jArray), 7);

	sum = s.sum<int, double>(jArray);
	mean = s.mean<int, double>(jArray);
	stddev = s.stddev<int, double>(jArray);

	EXPECT_DOUBLE_EQ(sum, 110.0);
	EXPECT_DOUBLE_EQ(mean, 5.5);
	EXPECT_DOUBLE_EQ(stddev, 2.9468984587725089);

	// test for normal array of data
	int kArray[] = {2, 3, 5, 7, 1, 6, 8, 10, 9, 4, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10};

	sum = s.sum<int, double>(kArray, 20);
	double sumAbs = s.sumAbs<int, double>(kArray, 20);

	EXPECT_DOUBLE_EQ(sum, 0.0);
	EXPECT_DOUBLE_EQ(sumAbs, 110.);
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Construct a new TEST object for data arrays with stride
 * 
 */
TEST(stats_arraywithstride, HandlesStatsforArraywithStride)
{
	// Create an instance of stats object
	umuq::stats s;

	// iArray is a two column array of size 10 * 2 = 20
	// we are interested to the first column so the Stride is 2
	int iArray[] = {2, 0,
					3, 0,
					5, 0,
					7, 0,
					1, 0,
					6, 0,
					8, 0,
					10, 0,
					9, 0,
					4, 0};

	EXPECT_EQ(s.minelement<int>(iArray, 20, 2), 1);
	EXPECT_EQ(s.minelement_index<int>(iArray, 20, 2), 8);
	EXPECT_EQ(s.maxelement<int>(iArray, 20, 2), 10);
	EXPECT_EQ(s.maxelement_index<int>(iArray, 20, 2), 14);

	// dArray is an array of size 4 * 3 = 12
	// We need to compute the sum of different columns
	double dArray[] = {1, 2, 3,
					   1, 2, 3,
					   1, 2, 3,
					   1, 2, 3};

	// Compute sum of column 0
	double sum = s.sum<double>(dArray, 12, 3);
	EXPECT_DOUBLE_EQ(sum, 4.0);

	// Compute sum of column 1
	sum = s.sum<double>(dArray + 1, 11, 3);
	EXPECT_DOUBLE_EQ(sum, 8.0);

	// Compute sum of column 2
	sum = s.sum<double>(dArray + 2, 10, 3);
	EXPECT_DOUBLE_EQ(sum, 12.0);
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Construct a new TEST object for testing median
 * 
 */
TEST(stats_test, HandlesMedianandMad)
{
	// Create an instance of stats object
	umuq::stats s;

	int iArray[] = {1, 1, 2, 2, 4, 6, 9};
	int med;
	int mad = s.medianAbs<int, int>(iArray, 7, 1, med);

	// The dataset has a median value of 2.
	EXPECT_EQ(med, 2);

	/*!
     * The absolute deviations about 2 in the data set are (1, 1, 0, 0, 2, 4, 7) which 
     * in turn have a median value of 1 (because the sorted absolute deviations are 
     * (0, 0, 1, 1, 2, 4, 7)). 
     * So the median absolute deviation for this data is 1.
     */
	EXPECT_EQ(mad, 1);

	// Check the std::vector
	std::vector<int> jArray(iArray, iArray + 7);
	mad = s.medianAbs<int, int>(jArray, med);

	EXPECT_EQ(med, 2);
	EXPECT_EQ(mad, 1);
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Construct a new TEST object for testing minmaxNormal
 * 
 */
TEST(stats_test, HandlesminmaxNormal)
{
	// Create an instance of stats object
	umuq::stats s;

	{ // Input array
		double iArray[] = {1.0, 2.0, 3.0, 4.0, 5.0};

		s.minmaxNormal<double>(iArray, 5);

		// Min max normalized array
		double nArray[] = {0.0, 0.25, 0.5, 0.75, 1.0};

		for (int i = 0; i < 5; i++)
		{
			EXPECT_DOUBLE_EQ(iArray[i], nArray[i]);
		}

		std::vector<double> jArray{1.0, 2.0, 3.0, 4.0, 5.0};

		s.minmaxNormal<double>(jArray);

		for (int i = 0; i < 5; i++)
		{
			EXPECT_DOUBLE_EQ(jArray[i], nArray[i]);
		}

		std::vector<double> kArray{1.0, 2.0, 3.0, 4.0, 5.0};

		s.minmaxNormal<double>(kArray, 0.0, 100.);

		// Min max normalized array
		double mArray[] = {0.01, 0.02, 0.03, 0.04, 0.05};

		for (int i = 0; i < 5; i++)
		{
			EXPECT_DOUBLE_EQ(kArray[i], mArray[i]);
		}
	}

	{ // Input array
		double iArray[] = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0};

		s.minmaxNormal<double>(iArray, 10, 2);

		// Min max normalized array
		double nArray[] = {0.0, 0.5, 1.0, 0.25, 0.75};

		for (int i = 0, j = 0; i < 10; i += 2, j++)
		{
			EXPECT_DOUBLE_EQ(iArray[i], nArray[j]);
		}
	}

	{ // Input array
		std::vector<double> iArray{1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0};

		s.minmaxNormal<double>(iArray.data(), 10, 2, 0.0, 10.0);

		// Min max normalized array
		double nArray[] = {0.1, 0.3, 0.5, 0.2, 0.4};

		for (int i = 0, j = 0; i < 10; i += 2, j++)
		{
			EXPECT_DOUBLE_EQ(iArray[i], nArray[j]);
		}
	}
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Construct a new TEST object for testing zscoreNormal
 * 
 */
TEST(stats_test, HandleszscoreNormal)
{
	// Create an instance of stats object
	umuq::stats s;

	{ // Input array
		double iArray[] = {1.0, 2.0, 3.0, 4.0, 5.0};

		s.zscoreNormal<double>(iArray, 5);

		// zscoreNormal normalized array
		double nArray[] = {-1.2649110640673518, -0.63245553203367588, 0.0, 0.63245553203367588, 1.2649110640673518};

		for (int i = 0; i < 5; i++)
		{
			EXPECT_DOUBLE_EQ(iArray[i], nArray[i]);
		}
	}

	{ // Input array
		double iArray[] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0};

		s.zscoreNormal<double>(iArray, 10, 2);

		//zscoreNormal normalized array with stride
		double nArray[] = {-1.2649110640673518, -0.63245553203367588, 0.0, 0.63245553203367588, 1.2649110640673518};

		for (int i = 0, j = 0; i < 10; i += 2, j++)
		{
			EXPECT_DOUBLE_EQ(iArray[i], nArray[j]);
		}
	}
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Construct a new TEST object for testing robustzscoreNormal
 * 
 */
TEST(stats_test, HandlesrobustzscoreNormal)
{
	// Create an instance of stats object
	umuq::stats s;

	{ // Input array
		double iArray[] = {1.0, 2.0, 3.0, 4.0, 5.0};

		s.robustzscoreNormal<double>(iArray, 5);

		// robustzscoreNormal normalized array
		double nArray[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

		for (int i = 0; i < 5; i++)
		{
			EXPECT_DOUBLE_EQ(iArray[i], nArray[i]);
		}
	}

	{ // Input array
		double iArray[] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0};

		s.robustzscoreNormal<double>(iArray, 10, 2);

		// robustzscoreNormal normalized array with stride
		double nArray[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

		for (int i = 0, j = 0; i < 10; i += 2, j++)
		{
			EXPECT_DOUBLE_EQ(iArray[i], nArray[j]);
		}
	}
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Construct a new TEST object for covariance
 * 
 */
TEST(stats_test, HandlesCovariance)
{
	// Create an instance of stats object
	umuq::stats s;

	// Create two vectors and compute their covariance.
	{
		double idata[] = {2.1, 2.5, 3.6, 4.0}; // (mean = 3.1)
		double jdata[] = {8, 10, 12, 14};	  // (mean = 11)

		double Covariance = s.covariance<double, double>(idata, jdata, 4);

		EXPECT_DOUBLE_EQ(Covariance, 6.8 / 3);
	}

	{
		std::vector<double> idata{2.1, 2.5, 3.6, 4.0}; // (mean = 3.1)
		std::vector<double> jdata{8, 10, 12, 14};	  // (mean = 11)

		double Covariance = s.covariance<double, double>(idata, jdata);

		EXPECT_DOUBLE_EQ(Covariance, 6.8 / 3);
	}

	// Create a 3-by-4 matrix and compute its covariance.
	{
		double idata1[] = {5, 0, 3, 7,
						   1, -5, 7, 3,
						   4, 9, 8, 10};

		// Covariance using pointers
		double *Covariance;

		// Compute the covariance
		Covariance = s.covariance<double, double>(idata1, 12, 4, 4);

		// Covariance computed with MATLAB
		double c1[] = {4.333333333333334, 8.833333333333332, -3.0, 5.666666666666667,
					   8.833333333333332, 50.333333333333336, 6.50, 24.166666666666668,
					   -3.0, 6.50, 7.0, 1.0,
					   5.666666666666667, 24.166666666666668, 1.0, 12.333333333333334};

		for (int i = 0; i < 12; i++)
		{
			EXPECT_DOUBLE_EQ(Covariance[i], c1[i]);
		}

		// Free memory
		delete[] Covariance;
		Covariance = nullptr;
	}

	// Create a 3-by-2 matrix and compute its covariance.
	{
		double idata2[] = {4.348817, 4.711934,
						   2.995049, 1.190864,
						   -3.793431, -1.357363};

		// Covariance using smart pointers
		std::unique_ptr<double[]> Covariance;

		// Compute the covariance
		Covariance.reset(std::move(s.covariance<double, double>(idata2, 6, 2, 2)));

		// Covariance computed with MATLAB
		double c2[] = {19.035391833621333, 11.913836879396001,
					   11.913836879396001, 9.287960143773001};

		for (int i = 0; i < 4; i++)
		{
			EXPECT_DOUBLE_EQ(Covariance[i], c2[i]);
		}
	}

	// Create a 3-by-2 matrix and compute its covariance using mean of each column(dimension).
	{
		double idata2[] = {4.348817, 4.711934,
						   2.995049, 1.190864,
						   -3.793431, -1.357363};

		// Covariance using smart pointers
		std::unique_ptr<double[]> Covariance;
		std::unique_ptr<double[]> Mean(new double[2]);
		Mean[0] = (idata2[0] + idata2[2] + idata2[4]) / 3.0;
		Mean[1] = (idata2[1] + idata2[3] + idata2[5]) / 3.0;

		// Compute the covariance
		Covariance.reset(std::move(s.covariance<double, double>(idata2, 6, 2, Mean.get())));

		// Covariance computed with MATLAB
		double c2[] = {19.035391833621333, 11.913836879396001,
					   11.913836879396001, 9.287960143773001};

		for (int i = 0; i < 4; i++)
		{
			EXPECT_DOUBLE_EQ(Covariance[i], c2[i]);
		}
	}
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Construct a new test for the unique member
 * 
 */
TEST(stats_test, HandlesunUniqueMemberFunctionality)
{
	// Vector of data which has some repetitive rows
	double p[] = {5, 12, 24,
				  12, 30, 59,
				  1, 4, 0,
				  0, -10, 1,
				  1, 2, 4,
				  2, 5, 10,
				  0, -1, -1,
				  1, 4, 0,
				  4, 25, -10,
				  0, -10, 1,
				  2, 5, 10,
				  1, 4, 0};

	// Array of unique row data (each row is unique)
	double pu[] = {5, 12, 24,
				   12, 30, 59,
				   1, 4, 0,
				   0, -10, 1,
				   1, 2, 4,
				   2, 5, 10,
				   0, -1, -1,
				   4, 25, -10};

	// Create an instance of stats object
	umuq::stats s;

	// vector
	std::vector<double> u;

	// Create a unique rows of data from p array
	s.unique<double>(p, 12, 3, u);

	// Check the size of unique data
	EXPECT_TRUE(u.size() == 24);

	// compare the elements
	for (std::size_t i = 0; i < u.size(); i++)
	{
		EXPECT_DOUBLE_EQ(u[i], pu[i]);
	}
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
