#include "io/io.hpp"
#include "gtest/gtest.h"

//! TEST for file existence
TEST(isFileExist_test, HandlesFiles)
{
    io f;
    EXPECT_TRUE(f.isFileExist("test.txt"));
    EXPECT_FALSE(f.isFileExist("utility.txt"));
};

//! TEST how IO handles files
TEST(openFile_test, HandlesFiles)
{
    //!An instance of io class
    io f;

    EXPECT_FALSE(f.isFileOpened());

    EXPECT_TRUE(f.openFile("test.txt"));
    EXPECT_TRUE(f.isFileOpened());
    int n = 0;
    while (f.readLine())
    {
        n++;
    }
    EXPECT_EQ(n, 82);
    f.closeFile();
    EXPECT_FALSE(f.isFileOpened());

    char *line = f.getLine();
    char **lineArg = f.getLineArg();

    EXPECT_EQ(NULL, line);
    EXPECT_EQ(NULL, lineArg);
};

//! TEST how IO handles std::fstream
TEST(openFilestream_test, HandlesFiles)
{
    //!An instance of io class
    io f;
    const char *fileName = "tmp";

    EXPECT_FALSE(f.openFile(fileName, f.in));
    EXPECT_FALSE(f.openFile(fileName, f.in | f.binary));
    EXPECT_TRUE(f.openFile(fileName, f.out));
    
    std::fstream &fs = f.getFstream();

    if (fs.is_open())
    {
        f.closeFile();
    }

    std::remove(fileName);
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}