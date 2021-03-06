#include "core/core.hpp"
#include "inference/tmcmc/runinfo.hpp"
#include "gtest/gtest.h"

// Tests parse
TEST(runinfo_test, HandlesConstruction)
{
    umuq::tmcmc::runinfo runinfoObj;

    EXPECT_EQ(0, runinfoObj.nDim);
    EXPECT_EQ(0, runinfoObj.maxGenerations);
    EXPECT_EQ(0, runinfoObj.currentGeneration);

    EXPECT_TRUE(runinfoObj.reset(2, 40));

    EXPECT_EQ(2, runinfoObj.nDim);
    EXPECT_EQ(40, runinfoObj.maxGenerations);

    for (int i = 0; i < runinfoObj.maxGenerations; i++)
    {
        if (i == 0)
        {
            EXPECT_DOUBLE_EQ(std::numeric_limits<double>::max(), runinfoObj.CoefVar[i]);
        }
        else
        {
            EXPECT_DOUBLE_EQ(0.0, runinfoObj.CoefVar[i]);
        }
        EXPECT_DOUBLE_EQ(0.0, runinfoObj.generationProbabilty[i]);
        EXPECT_EQ(0, runinfoObj.currentUniques[i]);
        EXPECT_DOUBLE_EQ(0.0, runinfoObj.logselection[i]);
        EXPECT_DOUBLE_EQ(0.0, runinfoObj.acceptance[i]);
    }
}

// Test IO functionality
TEST(runinfo_test, HandlesIO)
{
    std::remove("runinfo.txt");

    // Create an instance of runinfo object, initialize it
    // to some random value and save it to a file @"runinfo.txt"
    {
        umuq::tmcmc::runinfo runinfoObj(2, 10);

        EXPECT_EQ(2, runinfoObj.nDim);
        EXPECT_EQ(10, runinfoObj.maxGenerations);

        runinfoObj.currentGeneration = 9;

        for (int i = 0; i < runinfoObj.maxGenerations; i++)
        {
            runinfoObj.CoefVar[i] = static_cast<double>(i);
            runinfoObj.generationProbabilty[i] = static_cast<double>(i * i);
            runinfoObj.currentUniques[i] = i;
            runinfoObj.logselection[i] = static_cast<double>(i * i * i);
            runinfoObj.acceptance[i] = static_cast<double>(i) / 10.;
            runinfoObj.meantheta[i * runinfoObj.nDim] = static_cast<double>(i);
            runinfoObj.meantheta[i * runinfoObj.nDim + 1] = static_cast<double>(i);
        }
        runinfoObj.covariance[0] = 12.;
        runinfoObj.covariance[1] = 123.;
        runinfoObj.covariance[2] = 112.;
        runinfoObj.covariance[3] = 13.;

        EXPECT_TRUE(runinfoObj.save());
    }
    
    // Create an instance of runinfo object, initialize it
    // from a file @"runinfo.txt"
    {
        umuq::tmcmc::runinfo runinfoObj;

        EXPECT_TRUE(runinfoObj.load());
        EXPECT_EQ(2, runinfoObj.nDim);
        EXPECT_EQ(10, runinfoObj.maxGenerations);
        for (int i = 0; i < runinfoObj.maxGenerations; i++)
        {
            EXPECT_DOUBLE_EQ(static_cast<double>(i), runinfoObj.CoefVar[i]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i * i), runinfoObj.generationProbabilty[i]);
            EXPECT_EQ(i, runinfoObj.currentUniques[i]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i * i * i), runinfoObj.logselection[i]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i) / 10., runinfoObj.acceptance[i]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i), runinfoObj.meantheta[i * runinfoObj.nDim]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i), runinfoObj.meantheta[i * runinfoObj.nDim + 1]);
        }

        EXPECT_DOUBLE_EQ(12., runinfoObj.covariance[0]);
        EXPECT_DOUBLE_EQ(123., runinfoObj.covariance[1]);
        EXPECT_DOUBLE_EQ(112., runinfoObj.covariance[2]);
        EXPECT_DOUBLE_EQ(13., runinfoObj.covariance[3]);
    }

    std::remove("runinfo.txt");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
