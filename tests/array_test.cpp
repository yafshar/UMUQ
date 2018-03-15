#include <iostream>
#include <stdio.h>

#include "../src/misc/array.hpp"
#include "gtest/gtest.h"

TEST(ArrayWrapper, HandlesVectors)
{
    int *iarray;
    iarray = new int[1000];

    ArrayWrapper<int> it(iarray, 1000);
    for (auto i = it.begin(); i != it.end(); i++) {
        std::cout << *i << std::endl;
    }
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}