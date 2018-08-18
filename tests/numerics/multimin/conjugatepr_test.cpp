#include "core/core.hpp"
#include "numerics/multimin/conjugatepr.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check conjugatePr construction
 */
TEST(conjugatePr_test, HandlesMinimizerConstruction)
{
    conjugatePr<double> cPr;

}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
