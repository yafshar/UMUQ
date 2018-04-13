#include "core/core.hpp"
#include "numerics/stats.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check stats functionality
 */
TEST(stats_test, HandlesStats)
{
    stats s;
    int Iarray[] = {2, 3, 5, 7, 1, 6, 8, 10, 9, 4};
    EXPECT_EQ(s.minelement<int>(Iarray, 10), 1);
    EXPECT_EQ(s.maxelement<int>(Iarray, 10), 10);
    EXPECT_EQ(s.min_at<int>(Iarray, 10), 4);
    EXPECT_EQ(s.max_at<int>(Iarray, 10), 7);
    double sum = s.sum<int, double>(Iarray, 10);
    double mean = s.mean<int, double>(Iarray, 10);
    double stddev = s.stddev<int, double>(Iarray, 10);
    EXPECT_DOUBLE_EQ(sum, 55.0);
    EXPECT_DOUBLE_EQ(mean, 5.5);
    EXPECT_DOUBLE_EQ(stddev, 3.027650354097491);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
