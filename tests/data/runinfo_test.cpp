#include "core/core.hpp"
#include "data/runinfo.hpp"
#include "gtest/gtest.h"

//! Tests parse
TEST(runinfo_test, HandlesConstruction)
{
    umuq::tmcmc::runinfo<double> runinfoObj;

    EXPECT_EQ(0, runinfoObj.nDim);
    EXPECT_EQ(0, runinfoObj.maxGenerations);
    EXPECT_EQ(0, runinfoObj.Generation);

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
        EXPECT_DOUBLE_EQ(0.0, runinfoObj.p[i]);
        EXPECT_EQ(0, runinfoObj.currentuniques[i]);
        EXPECT_DOUBLE_EQ(0.0, runinfoObj.logselection[i]);
        EXPECT_DOUBLE_EQ(0.0, runinfoObj.acceptance[i]);
    }
}

//! Test IO functionality
TEST(runinfo_test, HandlesIO)
{
    std::remove("runinfo.txt");

    // Create an instance of runinfo object, initialize it
    // to some random value and save it to a file @"runinfo.txt"
    {
        umuq::tmcmc::runinfo<double> runinfoObj(2, 10);

        EXPECT_EQ(2, runinfoObj.nDim);
        EXPECT_EQ(10, runinfoObj.maxGenerations);

        runinfoObj.Generation = 9;

        for (int i = 0; i < runinfoObj.maxGenerations; i++)
        {
            runinfoObj.CoefVar[i] = static_cast<double>(i);
            runinfoObj.p[i] = static_cast<double>(i * i);
            runinfoObj.currentuniques[i] = i;
            runinfoObj.logselection[i] = static_cast<double>(i * i * i);
            runinfoObj.acceptance[i] = static_cast<double>(i) / 10.;
            runinfoObj.meantheta[i * runinfoObj.nDim] = static_cast<double>(i);
            runinfoObj.meantheta[i * runinfoObj.nDim + 1] = static_cast<double>(i);
        }
        runinfoObj.SS[0] = 12.;
        runinfoObj.SS[1] = 123.;
        runinfoObj.SS[2] = 112.;
        runinfoObj.SS[3] = 13.;

        EXPECT_TRUE(runinfoObj.save());
    }
    
    // Create an instance of runinfo object, initialize it
    // from a file @"runinfo.txt"
    {
        umuq::tmcmc::runinfo<double> runinfoObj;

        EXPECT_TRUE(runinfoObj.load());
        EXPECT_EQ(2, runinfoObj.nDim);
        EXPECT_EQ(10, runinfoObj.maxGenerations);
        for (int i = 0; i < runinfoObj.maxGenerations; i++)
        {
            EXPECT_DOUBLE_EQ(static_cast<double>(i), runinfoObj.CoefVar[i]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i * i), runinfoObj.p[i]);
            EXPECT_EQ(i, runinfoObj.currentuniques[i]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i * i * i), runinfoObj.logselection[i]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i) / 10., runinfoObj.acceptance[i]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i), runinfoObj.meantheta[i * runinfoObj.nDim]);
            EXPECT_DOUBLE_EQ(static_cast<double>(i), runinfoObj.meantheta[i * runinfoObj.nDim + 1]);
        }

        EXPECT_DOUBLE_EQ(12., runinfoObj.SS[0]);
        EXPECT_DOUBLE_EQ(123., runinfoObj.SS[1]);
        EXPECT_DOUBLE_EQ(112., runinfoObj.SS[2]);
        EXPECT_DOUBLE_EQ(13., runinfoObj.SS[3]);
    }

    std::remove("runinfo.txt");
}

//! Tests the unique member
TEST(database_uniquetest, HandlesunUniqueMemberFunctionality)
{
    //! Create an instance of a runinfo object
    umuq::tmcmc::runinfo<double> runinfoObj(3, 1);

    //! Vector of data which has some repetetive rows
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

    //! Array of unique row data (each row is unqiue)
    double pu[] = {5, 12, 24,
                   12, 30, 59,
                   1, 4, 0,
                   0, -10, 1,
                   1, 2, 4,
                   2, 5, 10,
                   0, -1, -1,
                   4, 25, -10};

    //! vector
    std::vector<double> u;

    //! Create a unique rows of data from p array
    runinfoObj.getUniques(p, 12, 3, u);

	//! Check the size of unique data
    EXPECT_TRUE(u.size() == 24);

	//! compqare the elements
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
