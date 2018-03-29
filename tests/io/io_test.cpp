#include "io/io.hpp"
#include "gtest/gtest.h"

//! TEST for file existence 
TEST(isFileExist_test, HandlesFiles)
{
    io out;
    EXPECT_TRUE(out.isFileExist("test.txt"));
    EXPECT_FALSE(out.isFileExist("utility.txt"));
};

//! TEST how IO handles files
TEST(openFile_test, HandlesFiles)
{
    //!An instance of io class
    io out;
    
    EXPECT_FALSE(out.isFileOpened());

    EXPECT_TRUE(out.openFile("test.txt"));
    EXPECT_TRUE(out.isFileOpened());
    int n = 0;
    while (out.readLine())
    {
        n++;
    }
    EXPECT_EQ(n, 82);
    out.closeFile();
    EXPECT_FALSE(out.isFileOpened());
    EXPECT_EQ(NULL, out.f);
    EXPECT_EQ(NULL, out.line);
    EXPECT_EQ(NULL, out.lineArg);
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}