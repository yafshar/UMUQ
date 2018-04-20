#include "core/core.hpp"
#include "numerics/factorial.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check factorial functionality
 */
TEST(factorial_test, HandlesInput)
{
    EXPECT_FLOAT_EQ(factorial<float>(0), 1.f);
    EXPECT_FLOAT_EQ(factorial<float>(1), 1.f);
    EXPECT_FLOAT_EQ(factorial<float>(6), 720.f);
    EXPECT_FLOAT_EQ(factorial<float>(34), 0.29523279903960414084761860964352e39f);

    EXPECT_DOUBLE_EQ(factorial<double>(0), 1.);
    EXPECT_DOUBLE_EQ(factorial<double>(1), 1.);
    EXPECT_DOUBLE_EQ(factorial<double>(10), 3628800.0);
    EXPECT_DOUBLE_EQ(factorial<double>(170), 0.7257415615307998967396728211129263114717e307);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
