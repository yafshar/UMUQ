#include <iostream>

#include "../src/misc/utility.hpp"
#include "gtest/gtest.h"

// Tests
TEST(eigen_test, HandlesMap){};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
