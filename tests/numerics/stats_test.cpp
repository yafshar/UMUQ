#include "core/core.hpp"
#include "numerics/stats.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check stats functionality
 */
TEST(stats_test, HandlesStats)
{
    //Create an instane of stats object
    stats s;

    int Iarray[] = {2, 3, 5, 7, 1, 6, 8, 10, 9, 4};

    EXPECT_EQ(s.minelement<int>(Iarray, 10), 1);
    EXPECT_EQ(s.maxelement<int>(Iarray, 10), 10);
    EXPECT_EQ(s.minelement_index<int>(Iarray, 10), 4);
    EXPECT_EQ(s.maxelement_index<int>(Iarray, 10), 7);

    double sum = s.sum<int, double>(Iarray, 10);
    double mean = s.mean<int, double>(Iarray, 10);
    double stddev = s.stddev<int, double>(Iarray, 10);

    EXPECT_DOUBLE_EQ(sum, 55.0);
    EXPECT_DOUBLE_EQ(mean, 5.5);
    EXPECT_DOUBLE_EQ(stddev, 3.027650354097491);
}

TEST(stats_arraywithstride, HandlesStatsforArraywithStride)
{
    //Create an instane of stats object
    stats s;

    //Iarray is a two column array of size 10 * 2 = 20 
    //we are interested to the first column so the Stride is 2
    int Iarray[] = {2, 0, 3, 0, 5, 0, 7, 0, 1, 0, 6, 0, 8, 0, 10, 0, 9, 0, 4, 0};

    EXPECT_EQ(s.minelement<int>(Iarray, 20, 2), 1);
    EXPECT_EQ(s.minelement_index<int>(Iarray, 20, 2), 8);
    EXPECT_EQ(s.maxelement<int>(Iarray, 20, 2), 10);
    EXPECT_EQ(s.maxelement_index<int>(Iarray, 20, 2), 14);

    //Darray is an array of size 4 * 3 = 12
    //We need to compute the sum of different columns
    double Darray[] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};

    //Compute sum of column 0
    double sum = s.sum<double>(Darray, 12, 3);
    EXPECT_DOUBLE_EQ(sum, 4.0);

    //Compute sum of column 1
    sum = s.sum<double>(Darray + 1, 11, 3);
    EXPECT_DOUBLE_EQ(sum, 8.0);
    
    // Compute sum of column 2
    sum = s.sum<double>(Darray + 2, 10, 3);
    EXPECT_DOUBLE_EQ(sum, 12.0);
}

TEST(stats_test, HandlesMedianandMad)
{
    //Create an instane of stats object
    stats s;

    int Iarray[] = {1, 1, 2, 2, 4, 6, 9};
    int med;
    int mad = s.medianAbs<int, int>(Iarray, 7, 1, med);

    //the dataset has a median value of 2.
    EXPECT_EQ(med, 2);

    /*!
     * The absolute deviations about 2 in the data set are (1, 1, 0, 0, 2, 4, 7) which 
     * in turn have a median value of 1 (because the sorted absolute deviations are 
     * (0, 0, 1, 1, 2, 4, 7)). 
     * So the median absolute deviation for this data is 1.
     */
    EXPECT_EQ(mad, 1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
