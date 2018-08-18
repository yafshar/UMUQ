#include "core/core.hpp"
#include "numerics/multimin/bfgs.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check bfgs construction
 */
TEST(bfgs_test, HandlesMinimizerConstruction)
{
    bfgs<double> bm;

}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
