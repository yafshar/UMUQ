#include "core/core.hpp"
#include "core/environment.hpp"
#include "misc/funcallcounter.hpp"
#include "gtest/gtest.h"

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
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}