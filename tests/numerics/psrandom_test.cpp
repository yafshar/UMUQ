#include "torc.h"
#include "core/core.hpp"
#include "numerics/eigenmatrix.hpp"
#include "numerics/psrandom.hpp"
#include "gtest/gtest.h"

class TORCEnvironment : public ::testing::Environment
{
  public:
	virtual void SetUp()
	{
		char **argv;
		int argc = 0;

		torc_init(argc, argv, 0);

		// ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
		// if (torc_node_id() != 0)
		// {
		//     delete listeners.Release(listeners.default_result_printer());
		// }
	}

	virtual void TearDown()
	{
		torc_finalize();
	}

	virtual ~TORCEnvironment() {}
};

/*! 
 * Test to check random functionality
 */
TEST(random_test, HandlesRandoms)
{
	psrandom r;
	EXPECT_TRUE(r.init());

	EMatrix2d e;
	e << 1, 3. / 5., 3. / 5., 2.;
	EVector2d d;
	d.Zero();
	std::cout << "e=" << e << std::endl;
	std::cout << "d=" << d << std::endl;

	mvnormdist<EMatrix2d, EVector2d> m(e);

	std::cout << "m=" << m() << std::endl;
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	::testing::AddGlobalTestEnvironment(new TORCEnvironment);

	return RUN_ALL_TESTS();
}
