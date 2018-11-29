#include "core/core.hpp"
#include "misc/timer.hpp"
#include "gtest/gtest.h"

// Test timer function
TEST(timer_test, Stopwatch)
{
    // Creat an instance of the timer object which would print output to a stream buffer
    umuq::umuqTimer t1;

    // Creat an instance of the timer object which would not print output to a stream buffer unless explicitly been asked to print
    umuq::umuqTimer t2(false);
    {
        int n = 100000;
        int s(0);
        t1.tic();
        for (int i = 0; i < n; i++)
        {
            s += (i + 1) + (i - 1);
        }
        t1.toc("Sum of two functions");
        EXPECT_EQ(s, (n - 1) * n);
    }
    t2.toc();

    t2.tic();
    {
        int n = 110000;
        int s(0);
        t1.tic();
        for (int i = 0; i < n; i++)
        {
            s += (i + 1) + (i - 1);
        }
        t1.toc("Sum of two functions");
        EXPECT_EQ(s, (n - 1) * n);
    }
    t2.toc();
    t2.print();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}