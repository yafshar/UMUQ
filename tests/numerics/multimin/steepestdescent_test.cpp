#include "core/core.hpp"
#include "numerics/multimin/steepestdescent.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check steepestDescent construction
 */
TEST(steepestDescent_test, HandlesMinimizerConstruction)
{
    steepestDescent<double> sdes;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
