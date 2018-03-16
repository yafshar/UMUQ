#include <iostream>

#include "../src/misc/array.hpp"
#include "gtest/gtest.h"

TEST(ArrayWrapper, HandlesVectors)
{
    int *iarray;
    iarray = new int[10];

    for (int j = 0; j < 10; j++)
        iarray[j] = j * 10;

    ArrayWrapper<int> it(iarray, 10);
    for (auto i = it.begin(); i != it.end(); i++)
    {
        std::cout << *i << std::endl;
    }
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}