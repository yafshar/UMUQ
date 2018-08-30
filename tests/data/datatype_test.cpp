#include "core/core.hpp"
#include "environment.hpp"
#include "data/datatype.hpp"
#include "gtest/gtest.h"

//! Tests datatype which is using database object
TEST(datatype_test, HandlesGlobalData)
{
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment<>);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new UMUQEventListener);

    return RUN_ALL_TESTS();
}