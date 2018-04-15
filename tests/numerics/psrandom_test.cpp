#include "torc.h"
#include "core/core.hpp"
#include "numerics/eigenmatrix.hpp"
#include "numerics/psrandom.hpp"
#include "gtest/gtest.h"

psrandom r;

/*! 
 * Test to check random functionality
 */
TEST(random_test, HandlesRandoms)
{
	EXPECT_TRUE(r.init());
}

int main(int argc, char **argv)
{
	torc_register_task((void *)r.init_Task);
	torc_init(argc, argv, 0);

	::testing::InitGoogleTest(&argc, argv);
	int res = RUN_ALL_TESTS();

	torc_finalize();

	return res;
}
