#include "core/core.hpp"
#include "environment.hpp"
#include "misc/funcallcounter.hpp"
#include "gtest/gtest.h"

/*!
 * \brief In this test, it is important to register the task before calling the torcEnvironment
 * 
 */

//! Create an instance of funcallcounter object
umuq::funcallcounter fc;

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

    // Number of tasks
    int const ntasks = 100;

    for (int i = 0; i < ntasks; i++)
    {
        torc_create(-1, (void (*)())taskf, 0);
    }
    torc_waitall();

    fc.count();

    EXPECT_LE(fc.getLocalFunctionCallsNumber(), ntasks);

    //! Reset the local counter to zero
    fc.reset();

    EXPECT_EQ(fc.getLocalFunctionCallsNumber(), 0);

    EXPECT_EQ(fc.getGlobalFunctionCallsNumber(), ntasks);
    EXPECT_EQ(fc.getTotalFunctionCallsNumber(), ntasks);

    for (int i = 0; i < ntasks * 2; i++)
    {
        torc_create(-1, (void (*)())taskf, 0);
    }
    torc_waitall();

    fc.count();

    fc.reset();

    EXPECT_EQ(fc.getGlobalFunctionCallsNumber(), ntasks * 2);

    EXPECT_EQ(fc.getTotalFunctionCallsNumber(), ntasks * 3);
}

int main(int argc, char **argv)
{
    //! First, we register the task
    torc_register_task((void *)taskf);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment<>);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}