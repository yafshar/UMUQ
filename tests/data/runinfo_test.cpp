#include "core/core.hpp"
#include "data/runinfo.hpp"
#include "gtest/gtest.h"

//! Tests parse
TEST(runinfo_test, HandlesConstruction)
{
    runinfo<double> r;

    EXPECT_EQ(0, r.nDim);
    EXPECT_EQ(0, r.maxGenerations);
    EXPECT_EQ(0, r.Generation);

    EXPECT_TRUE(r.reset(2, 40));

    EXPECT_EQ(2, r.nDim);
    EXPECT_EQ(40, r.maxGenerations);

    for (int i = 0; i < r.maxGenerations; i++)
    {
        if (i == 0)
        {
            EXPECT_DOUBLE_EQ(std::numeric_limits<double>::max(), r.CoefVar[i]);
        }
        else
        {
            EXPECT_DOUBLE_EQ(0.0, r.CoefVar[i]);
        }
        EXPECT_DOUBLE_EQ(0.0, r.p[i]);
        EXPECT_EQ(0, r.currentuniques[i]);
        EXPECT_DOUBLE_EQ(0.0, r.logselection[i]);
        EXPECT_DOUBLE_EQ(0.0, r.acceptance[i]);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
