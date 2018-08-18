#include "core/core.hpp"
#include "numerics/multimin/simplexnm2.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check nmsimplex construction
 */
TEST(simplexNM2_test, HandlesMinimizerConstruction)
{
    simplexNM2<double> snm;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
