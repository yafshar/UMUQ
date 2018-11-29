#include "core/core.hpp"
#include "environment.hpp"
#include "inference/tmcmc/database.hpp"
#include "gtest/gtest.h"

// Tests database construction
TEST(database_test, HandlesConstruction)
{
    // Create an instance of database object
    {
        umuq::tmcmc::database d;

        EXPECT_EQ(0, d.nDimSamplePoints);
        EXPECT_EQ(0, d.nDimDataArray);
        EXPECT_EQ(std::size_t{}, d.idxPosition);
        EXPECT_EQ(0, d.nSamplePoints);
    }

    std::remove("database_100.txt");

    // Create an instance of database object
    {
        umuq::tmcmc::database d(2, 2, 3);

        EXPECT_EQ(2, d.nDimSamplePoints);
        EXPECT_EQ(2, d.nDimDataArray);
        EXPECT_EQ(std::size_t{}, d.idxPosition);
        EXPECT_EQ(3, d.nSamplePoints);

        {
            double p[] = {1., -1.};
            double g[] = {120., 321.};
            d.update(p, 1000., g, 1);
        }

        {
            double p[] = {2, 3.4};
            double g[] = {1206., 3621.};
            d.update(p, 10000., g, 1);
        }

        {
            double p[] = {4., 14};
            double g[] = {506., 132621.};
            d.update(p, 2000, g, 0);
        }

        EXPECT_DOUBLE_EQ(1., d.samplePoints[0]);
        EXPECT_DOUBLE_EQ(-1., d.samplePoints[1]);
        EXPECT_DOUBLE_EQ(2., d.samplePoints[2]);
        EXPECT_DOUBLE_EQ(3.4, d.samplePoints[3]);
        EXPECT_DOUBLE_EQ(4., d.samplePoints[4]);
        EXPECT_DOUBLE_EQ(14., d.samplePoints[5]);

        EXPECT_DOUBLE_EQ(120., d.dataArray[0]);
        EXPECT_DOUBLE_EQ(321., d.dataArray[1]);
        EXPECT_DOUBLE_EQ(1206., d.dataArray[2]);
        EXPECT_DOUBLE_EQ(3621, d.dataArray[3]);
        EXPECT_DOUBLE_EQ(506., d.dataArray[4]);
        EXPECT_DOUBLE_EQ(132621., d.dataArray[5]);

        EXPECT_DOUBLE_EQ(1000., d.fValue[0]);
        EXPECT_DOUBLE_EQ(10000., d.fValue[1]);
        EXPECT_DOUBLE_EQ(2000., d.fValue[2]);

        EXPECT_EQ(1, d.surrogate[0]);
        EXPECT_EQ(1, d.surrogate[1]);
        EXPECT_EQ(0, d.surrogate[2]);

        EXPECT_TRUE(d.save("database", 100));
    }

    {
        umuq::tmcmc::database d;
        EXPECT_FALSE(d.load("database", 100));
    }

    {
        umuq::tmcmc::database d(2, 2, 3);
        EXPECT_TRUE(d.load("database", 100));

        EXPECT_DOUBLE_EQ(1., d.samplePoints[0]);
        EXPECT_DOUBLE_EQ(-1., d.samplePoints[1]);
        EXPECT_DOUBLE_EQ(2., d.samplePoints[2]);
        EXPECT_DOUBLE_EQ(3.4, d.samplePoints[3]);
        EXPECT_DOUBLE_EQ(4., d.samplePoints[4]);
        EXPECT_DOUBLE_EQ(14., d.samplePoints[5]);

        EXPECT_DOUBLE_EQ(120., d.dataArray[0]);
        EXPECT_DOUBLE_EQ(321., d.dataArray[1]);
        EXPECT_DOUBLE_EQ(1206., d.dataArray[2]);
        EXPECT_DOUBLE_EQ(3621, d.dataArray[3]);
        EXPECT_DOUBLE_EQ(506., d.dataArray[4]);
        EXPECT_DOUBLE_EQ(132621., d.dataArray[5]);

        EXPECT_DOUBLE_EQ(1000., d.fValue[0]);
        EXPECT_DOUBLE_EQ(10000., d.fValue[1]);
        EXPECT_DOUBLE_EQ(2000., d.fValue[2]);
    }

    std::remove("database_100.txt");
}

// Tests datatype which is using database object
TEST(database_test, HandlesTask)
{
    {
        //Create an instance of a database object
        umuq::tmcmc::database d(2, 2, 3);

        //Update the data using different threads
        {
            double p[] = {1., -1.};
            double g[] = {120., 321.};
            d.update(p, 1000., g, 1);
        }

        {
            double p[] = {2, 3.4};
            double g[] = {1206., 3621.};
            d.update(p, 10000., g, 1);
        }

        {
            double p[] = {4., 14};
            double g[] = {506., 132621.};
            d.update(p, 2000, g, 0);
        }

        EXPECT_DOUBLE_EQ(1., d.samplePoints[0]);
        EXPECT_DOUBLE_EQ(-1., d.samplePoints[1]);
        EXPECT_DOUBLE_EQ(2., d.samplePoints[2]);
        EXPECT_DOUBLE_EQ(3.4, d.samplePoints[3]);
        EXPECT_DOUBLE_EQ(4., d.samplePoints[4]);
        EXPECT_DOUBLE_EQ(14., d.samplePoints[5]);

        EXPECT_DOUBLE_EQ(120., d.dataArray[0]);
        EXPECT_DOUBLE_EQ(321., d.dataArray[1]);
        EXPECT_DOUBLE_EQ(1206., d.dataArray[2]);
        EXPECT_DOUBLE_EQ(3621, d.dataArray[3]);
        EXPECT_DOUBLE_EQ(506., d.dataArray[4]);
        EXPECT_DOUBLE_EQ(132621., d.dataArray[5]);

        EXPECT_DOUBLE_EQ(1000., d.fValue[0]);
        EXPECT_DOUBLE_EQ(10000., d.fValue[1]);
        EXPECT_DOUBLE_EQ(2000., d.fValue[2]);

        EXPECT_EQ(1, d.surrogate[0]);
        EXPECT_EQ(1, d.surrogate[1]);
        EXPECT_EQ(0, d.surrogate[2]);

        EXPECT_TRUE(d.save("database", 100));
    }

    std::remove("database_100.txt");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}