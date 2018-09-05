#include "core/core.hpp"
#include "misc/timer.hpp"
#include "gtest/gtest.h"

int f1(int const i)
{
    return i + 1;
}

int f2(int const i)
{
    return i - 1;
}

// Test timer function
TEST(timer_test, Stopwatch)
{
    int n = 100000;
    int s(0);
    umuq::UMUQTimer t;
    for (int i = 0; i < n; i++)
    {
        s += f1(i) + f2(i);
    }
    t.toc("Sum of two functions");
    EXPECT_EQ(s, (n - 1) * n);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}