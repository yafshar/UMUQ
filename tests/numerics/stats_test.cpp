#include "core/core.hpp"
#include "io/io.hpp"
#include "numerics/stats.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check stats functionality
 */
TEST(stats_test, HandlesStats)
{
    // Create an instane of stats object
    stats s;

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
}

/*!
 * \brief Construct a new TEST object for data arrays with stride
 * 
 */
TEST(stats_arraywithstride, HandlesStatsforArraywithStride)
{
    // Create an instane of stats object
    stats s;

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
 * \brief Construct a new TEST object for testing median
 * 
 */
TEST(stats_test, HandlesMedianandMad)
{
    // Create an instane of stats object
    stats s;

    int iArray[] = {1, 1, 2, 2, 4, 6, 9};
    int med;
    int mad = s.medianAbs<int, int>(iArray, 7, 1, med);

    // the dataset has a median value of 2.
    EXPECT_EQ(med, 2);

    /*!
     * The absolute deviations about 2 in the data set are (1, 1, 0, 0, 2, 4, 7) which 
     * in turn have a median value of 1 (because the sorted absolute deviations are 
     * (0, 0, 1, 1, 2, 4, 7)). 
     * So the median absolute deviation for this data is 1.
     */
    EXPECT_EQ(mad, 1);
}

/*!
 * \brief Construct a new TEST object for covariance
 * 
 */
TEST(stats_test, HandlesCovariance)
{
    // Create an instane of stats object
    stats s;

    // Create two vectors and compute their covariance.
    {
        double idata[] = {2.1, 2.5, 3.6, 4.0}; // (mean = 3.1)
        double jdata[] = {8, 10, 12, 14};      // (mean = 11)

        double Covariance = s.covariance<double, double>(idata, jdata, 4);

        EXPECT_DOUBLE_EQ(Covariance, 6.8 / 3);
    }

    // Create a 3-by-4 matrix and compute its covariance.
    double idata1[] = {5, 0, 3, 7,
                       1, -5, 7, 3,
                       4, 9, 8, 10};

    // Compute the covariance
    double *Covariance = nullptr;
    Covariance = s.covariance<double, double>(idata1, 12, 4, 4);
    if (Covariance)
    {
        // Covariance computed with MATLAB
        double c1[] = {4.333333333333334, 8.833333333333332, -3.0, 5.666666666666667,
                       8.833333333333332, 50.333333333333336, 6.50, 24.166666666666668,
                       -3.0, 6.50, 7.0, 1.0,
                       5.666666666666667, 24.166666666666668, 1.0, 12.333333333333334};

        for (int i = 0; i < 12; i++)
        {
            EXPECT_DOUBLE_EQ(Covariance[i], c1[i]);
        }

        delete[] Covariance;
        Covariance = nullptr;
    }

    double idata2[] = {4.348817, 4.711934,
                       2.995049, 1.190864,
                       -3.793431, -1.357363};

    // Compute the covariance
    Covariance = s.covariance<double, double>(idata2, 6, 2, 2);
    if (Covariance)
    {
        // Covariance computed with MATLAB
        double c2[] = {19.035391833621333,
                       11.913836879396001,
                       11.913836879396001,
                       9.287960143773001};

        for (int i = 0; i < 4; i++)
        {
            EXPECT_DOUBLE_EQ(Covariance[i], c2[i]);
        }

        delete[] Covariance;
        Covariance = nullptr;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
