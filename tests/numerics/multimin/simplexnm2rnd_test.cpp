#include "core/core.hpp"
#include "numerics/multimin/simplexnm2rnd.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check simplexNM2Rnd construction
 */
TEST(simplexNM2Rnd_test, HandlesMinimizerConstruction)
{
    simplexNM2Rnd<double> snmr;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
