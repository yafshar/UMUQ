#include "torc.h"
#include "core/core.hpp"
#include "misc/funcallcounter.hpp"
#include "gtest/gtest.h"

funcallcounter f;

/*! 
 * Test to check random functionality
 */
TEST(funcallcounter_test, HandlesFunctioncounter)
{
}

int main(int argc, char **argv)
{
	torc_register_task((void *)f.reset_Task);
	torc_register_task((void *)f.get_Task);

	torc_init(argc, argv, 0);

	::testing::InitGoogleTest(&argc, argv);
	int res = RUN_ALL_TESTS();

	torc_finalize();

	return res;
}