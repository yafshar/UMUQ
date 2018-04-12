#include "core/core.hpp"
#include "data/runinfo.hpp"
#include "gtest/gtest.h"

//! Tests parse
TEST(runinfo_test, HandlesConstruction)
{
    runinfo_t r;

    EXPECT_EQ(0, r.Gen);
    EXPECT_EQ(0, r.probdim);
    EXPECT_EQ(0, r.maxgens);

    r.init(2, 40);
    EXPECT_EQ(2, r.probdim);
    EXPECT_EQ(40, r.maxgens);

    for (int i = 0; i < r.maxgens; i++)
    {
        EXPECT_DOUBLE_EQ(0.0, r.p[i]);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
