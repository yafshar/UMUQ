#include "core/core.hpp"
#include "misc/parser.hpp"
#include "gtest/gtest.h"

#define BUFLEN 256

/*! Tests parse class in case of empty line or line with tabs and spaces
 *
 */
TEST(parse_test, HandlesZeroInput)
{
    parser p;

    p.parse("     ");

    std::string word = p.at<std::string>(0);

    EXPECT_EQ(word, "");

    //!checking for tab and next line
    p.parse("\t  \n");

    word = p.at<std::string>(0);

    EXPECT_EQ(word, "");
}

/*! 
 * Tests parse class to make sure it can parse commands correctly
 */
TEST(parse_test, HandlesInput)
{
    parser p;

    char line[BUFLEN];

    sprintf(line, "sh doall.sh");
    p.parse(line);

    std::string word[4];
    word[0] = "sh";
    word[1] = "doall.sh";

    for (int i = 0; i < 2; i++)
    {
        EXPECT_EQ(word[i], p.at<std::string>(i));
    }

    //!testing space at the start of line
    sprintf(line, "   sh doall.sh");
    p.parse(line);
    for (int i = 0; i < 2; i++)
    {
        EXPECT_EQ(word[i], p.at<std::string>(i));
    }

    sprintf(line, "bash doall.sh out 2>&1");
    p.parse(line);

    word[0] = "bash";
    word[1] = "doall.sh";
    word[2] = "out";
    word[3] = "2>&1";

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(word[i], p.at<std::string>(i));
    }
}

/*! 
 * Tests parse class in translating the text file
 */
TEST(parse_cmd, HandlesCmd)
{
    parser p;

    char line[BUFLEN];

    int n = 4;
    while (n--)
    {
        sprintf(line, "B%d              %d.05           10%d.012", n, n, n);
        p.parse(line);

        EXPECT_EQ(p.at<std::string>(0), "B" + std::to_string(n));
        EXPECT_DOUBLE_EQ(p.at<double>(1), 0.05 + (double)n);
        EXPECT_DOUBLE_EQ(p.at<double>(2), 100.012 + (double)n);
    }

    n = 8;
    while (n--)
    {
        sprintf(line, " PopSize 5000%d ", n);
        p.parse(line);
        EXPECT_EQ(p.at<std::string>(0), "PopSize");
        EXPECT_EQ(p.at<int>(1), 50000 + n);
    }

    sprintf(line, "   prior_mu    1.0,0.1 ");
    p.parse(line);

    EXPECT_DOUBLE_EQ(p.at<double>(1), 1.0);
    EXPECT_DOUBLE_EQ(p.at<double>(2), 0.1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}