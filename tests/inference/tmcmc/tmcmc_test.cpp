#include "core/core.hpp"
#include "core/environment.hpp"
#include "inference/tmcmc/tmcmc.hpp"
#include "gtest/gtest.h"

//! Tests tmcmc
TEST(tmcmc_test, HandlesConstruction)
{
    tmcmc<double> t;
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