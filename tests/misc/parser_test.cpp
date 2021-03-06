#include "core/core.hpp"
#include "misc/parser.hpp"
#include "gtest/gtest.h"

#define BUFLEN 256

/*!
 * \ingroup Test_Module
 *
 * \brief Tests parse class in case of empty line or line with tabs and spaces
 *
 */
TEST(parse_test, HandlesZeroInput)
{
    umuq::parser p;

    p.parse("     ");

    std::string word = p.at<std::string>(0);

    EXPECT_EQ(word, "");

    // Checking for tab and next line and return
    p.parse("\t  \n \r");

    word = p.at<std::string>(0);

    EXPECT_EQ(word, "");
}

/*!
 * \ingroup Test_Module
 *
 * \brief Tests parse class to make sure it can parse commands correctly
 */
TEST(parse_test, HandlesInput)
{
    umuq::parser p;

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

    // testing space at the start of line
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
 * \ingroup Test_Module
 *
 * \brief Tests parse class in translating the text file
 */
TEST(parse_cmd, HandlesCmd)
{
    umuq::parser p;

    char line[BUFLEN];

    int n = 4;
    while (n--)
    {
        sprintf(line, "B%d              %d.05           10%d.012", n, n, n);
        p.parse(line);

        EXPECT_EQ(p.at<std::string>(0), "B" + std::to_string(n));
        EXPECT_DOUBLE_EQ(p.at<double>(1), 0.05 + static_cast<double>(n));
        EXPECT_DOUBLE_EQ(p.at<double>(2), 100.012 + static_cast<double>(n));
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

    EXPECT_EQ(p.toupper(p.at<std::string>(0)), "PRIOR_MU");
    EXPECT_EQ(p.toupper(p.at<std::string>(1)), "1.0,0.1");

    EXPECT_DOUBLE_EQ(p.at<double>(1), 1.0);
    EXPECT_DOUBLE_EQ(p.at<double>(2), 0.1);
}

/*!
 * \ingroup Test_Module
 *
 * \brief Tests parse class lower and upper functiinality
 */
TEST(parse_test, HandlesUpperLowerCase)
{
    umuq::parser p;
    EXPECT_EQ(p.toupper("SUM_squared"), "SUM_SQUARED");
    EXPECT_EQ(p.toupper("sum_squared"), "SUM_SQUARED");
    EXPECT_EQ(p.toupper("sum_squared", 0, 1), "Sum_squared");
    EXPECT_EQ(p.toupper("sum_squared", 4, 5), "sum_Squared");
    EXPECT_EQ(p.tolower("SUM_squared"), "sum_squared");
    EXPECT_EQ(p.tolower("SUM_SQUARED"), "sum_squared");
    EXPECT_EQ(p.tolower("SUM_SQUARED", 0, 1), "sUM_SQUARED");
    EXPECT_EQ(p.tolower("SUM_SQUARED", 4, 5), "SUM_sQUARED");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
