#include "core/core.hpp"
#include "numerics/multimin/bfgs2.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check bfgs2 construction
 */
TEST(bfgs2_test, HandlesMinimizerConstruction)
{
    bfgs2<double> bm;

}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
