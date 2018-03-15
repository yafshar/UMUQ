#include <iostream>
#include <stdio.h>

#include "../src/misc/utility.hpp"
#include "gtest/gtest.h"

#define BUFLEN 1024

TEST(isFileExist_test, HandlesFiles)
{
    utility u;
    EXPECT_TRUE(u.isFileExist("test.txt"));
    EXPECT_FALSE(u.isFileExist("utility.txt"));
};

TEST(openFile_test, HandlesFiles)
{
    utility u;
    EXPECT_FALSE(u.isFileOpened());

    EXPECT_TRUE(u.openFile("test.txt"));
    EXPECT_TRUE(u.isFileOpened());
    int n = 0;
    while (u.readLine())
    {
        n++;
    }
    EXPECT_EQ(n, 82);
    u.closeFile();
    EXPECT_FALSE(u.isFileOpened());
    EXPECT_EQ(NULL, u.f);
    EXPECT_EQ(NULL, u.line);
    EXPECT_EQ(NULL, u.lineArg);

};

// Tests
TEST(execute_cmd_test, HandlesZeroInput){};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}