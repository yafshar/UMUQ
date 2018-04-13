#include "core/core.hpp"
#include "numerics/psrandom.hpp"
#include "gtest/gtest.h"

#define MODE_MS 0

/*! 
 * Test to check random functionality
 */
TEST(random_test, HandlesRandoms)
{
}

int main(int argc, char **argv)
{
    torc_init(argc, argv, MODE_MS);

    // torc_register_task((void *)r.mt19937_Init_task);
    // torc_register_task((void *)r.Saru_Init_task);

    psrandom r(280675);

    ::testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();

    torc_finalize();

    return res;
}
