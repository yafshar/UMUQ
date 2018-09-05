#include "core/core.hpp"
#include "environment.hpp"
#include "inference/tmcmc/tmcmc.hpp"
#include "gtest/gtest.h"

//! Tests tmcmc
TEST(tmcmc_test, HandlesConstruction)
{
    //! Create an instance of the tmcmc object
    umuq::tmcmc<double> t;

    //! Set the input file
    EXPECT_TRUE(t.setInputFileName("./data/test.txt"));

    //! Initilize the object
    EXPECT_TRUE(t.init());
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment<>);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}