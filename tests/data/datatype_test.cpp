#include "core/core.hpp"
#include "core/environment.hpp"
#include "data/datatype.hpp"
#include "gtest/gtest.h"

//! Tests datatype which is using database object
TEST(datatype_test, HandlesGlobalData)
{
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}