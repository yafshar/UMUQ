#include "core/core.hpp"
#include "numerics/multimin/conjugatefr.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check conjugateFr construction
 */
TEST(conjugateFr_test, HandlesMinimizerConstruction)
{
    conjugateFr<double> cFr;

}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
