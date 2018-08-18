#include "core/core.hpp"
#include "numerics/multimin/simplexnm.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check nmsimplex construction
 */
TEST(simplexNM_test, HandlesMinimizerConstruction)
{
    simplexNM<double> snm;

}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
