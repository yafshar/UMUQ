#include <iostream>

#include "../src/misc/utility.hpp"
#include "gtest/gtest.h"

// Tests
TEST(execute_cmd_test, HandlesZeroInput){};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}