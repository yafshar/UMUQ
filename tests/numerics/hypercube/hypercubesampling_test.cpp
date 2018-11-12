#include "core/core.hpp"
#include "environment.hpp"
#include "io/io.hpp"
#include "numerics/hypercube/hypercubesampling.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 * 
 * \brief Get an instance of a seeded double random object
 * 
 */
umuq::psrandom<double> prng(123);

/*!
 * \ingroup Test_Module
 *
 * Test to check hypercubeSampling construction
 */
TEST(hypercubeSamplingTest, HandlesConstruction)
{
    {
        // Problem dimension
        int numDimensions = 1;

        // Number of data points
        int const numDataPoints = 10;

        // Number of points in each direction
        std::vector<int> numPointsInEachDirection{10};

        // Create an instance of the hypercubeSampling object in unit hypercube
        umuq::hypercubeSampling<double> Domain(numPointsInEachDirection);

        std::vector<double> dataPoints(numDataPoints * numDimensions);

        // Create input points
        EXPECT_TRUE(Domain.grid(dataPoints));

        for (int i = 0; i < numDataPoints; i++)
        {
            EXPECT_DOUBLE_EQ(dataPoints[i], i * 1. / (numDataPoints - 1));
        }

        // Create input points in the middle of the cells
        EXPECT_TRUE(Domain.grid(dataPoints, 0));

        for (int i = 0; i < numDataPoints; i++)
        {
            EXPECT_DOUBLE_EQ(dataPoints[i], 0.5 / numDataPoints + i * 1. / numDataPoints);
        }
    }

    {
        // Problem dimension
        int numDimensions = 2;

        // Number of data points
        int const numDataPoints = 20;

        // Number of points in each direction
        std::vector<int> numPointsInEachDirection{4, 5};

        std::vector<double> LowerBound{0., 0.};
        std::vector<double> UpperBound{.6, 8.};

        // Create an instance of the hypercubeSampling object in unit hypercube
        umuq::hypercubeSampling<double> Domain(numPointsInEachDirection, LowerBound, UpperBound);

        std::vector<double> dataPoints(numDataPoints * numDimensions);
        std::vector<double> exactDataPoints{0, 0,
                                            0, 2,
                                            0, 4,
                                            0, 6,
                                            0, 8,
                                            0.2, 0,
                                            0.2, 2,
                                            0.2, 4,
                                            0.2, 6,
                                            0.2, 8,
                                            0.4, 0,
                                            0.4, 2,
                                            0.4, 4,
                                            0.4, 6,
                                            0.4, 8,
                                            0.6, 0,
                                            0.6, 2,
                                            0.6, 4,
                                            0.6, 6,
                                            0.6, 8};

        // Create input points
        EXPECT_TRUE(Domain.grid(dataPoints));

        for (auto dataIt = dataPoints.begin(), exactDataIt = exactDataPoints.begin(); dataIt != dataPoints.end(); dataIt++, exactDataIt++)
        {
            EXPECT_DOUBLE_EQ(*dataIt, *exactDataIt);
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment<>);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}
