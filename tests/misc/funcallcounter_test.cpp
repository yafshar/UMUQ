#include "torc.h"
#include "core/core.hpp"
#include "misc/funcallcounter.hpp"
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

funcallcounter f;

void taskf()
{
    f.increment();
}

/*! 
 * Test to check random functionality
 */
TEST(funcallcounter_test, HandlesFunctioncounter)
{
    EXPECT_TRUE(f.init());
    torc_register_task((void *)taskf);

    //Number of nodes
    int nnodes = torc_num_nodes();
    int ntasks = 100;

    for (int i = 0; i < ntasks; i++)
    {
        torc_create(-1, (void (*)())taskf, 0);
    }
    torc_waitall();

    f.count();
    f.reset();

    EXPECT_EQ(f.get_nglobalfc(), nnodes * ntasks);
    EXPECT_EQ(f.get_ntotalfc(), nnodes * ntasks);

    for (int i = 0; i < ntasks * 2; i++)
    {
        torc_create(-1, (void (*)())taskf, 0);
    }
    torc_waitall();

    f.count();
    f.reset();

    EXPECT_EQ(f.get_nglobalfc(), nnodes * ntasks * 2);
    EXPECT_EQ(f.get_ntotalfc(), nnodes * ntasks * 3);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new TORCEnvironment);

    return RUN_ALL_TESTS();
}