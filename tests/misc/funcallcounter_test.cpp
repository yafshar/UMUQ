#include "core/core.hpp"
#include "core/environment.hpp"
#include "misc/funcallcounter.hpp"
#include "gtest/gtest.h"

//! Create an instance of funcallcounter object
funcallcounter fc;

//! Global task
void taskf()
{
    fc.increment();
}

/*! 
 * Test to check Function call counters functionality
 */
TEST(funcallcounter_test, HandlesFunctioncounter)
{
    EXPECT_TRUE(fc.init());
    torc_register_task((void *)taskf);

    // Number of nodes
    int const nnodes = torc_num_nodes();
    int const ntasks = 100;

    for (int i = 0; i < ntasks; i++)
    {
        torc_create(-1, (void (*)())taskf, 0);
    }
    torc_waitall();

    fc.count();

    EXPECT_EQ(fc.getLocalFunctionCallsNumber(), nnodes * ntasks);
    
    //! Reset the local counter to zero
    fc.reset();

    EXPECT_EQ(fc.getLocalFunctionCallsNumber(), 0);
    EXPECT_EQ(fc.getGlobalFunctionCallsNumber(), nnodes * ntasks);
    EXPECT_EQ(fc.getTotalFunctionCallsNumber(), nnodes * ntasks);

    for (int i = 0; i < ntasks * 2; i++)
    {
        torc_create(-1, (void (*)())taskf, 0);
    }
    torc_waitall();

    fc.count();
    fc.reset();

    EXPECT_EQ(fc.getGlobalFunctionCallsNumber(), nnodes * ntasks * 2);
    EXPECT_EQ(fc.getTotalFunctionCallsNumber(), nnodes * ntasks * 3);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new UMUQEventListener);

    return RUN_ALL_TESTS();
}