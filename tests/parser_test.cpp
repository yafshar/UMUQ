#include <iostream>
#include <stdio.h>

#include "../src/misc/parser.hpp"
#include "gtest/gtest.h"

#define BUFLEN 1024

/*! Tests parse class in case of empty line or line with tabs and spaces
 *
 */
TEST(parse_test, HandlesZeroInput)
{
    parser p;

    char line[BUFLEN];
    char *largv[BUFLEN];

    sprintf(line, " ");
    p.parse(line, largv);

    const char *targv[0];
    targv[0] = "";

    EXPECT_STREQ(largv[0], targv[0]);

    //checking for tab and next line
    sprintf(line, "\t  \n");

    p.parse(line, largv);
    EXPECT_STREQ(largv[0], targv[0]);
};

/*! 
 * Tests parse class to make sure it can parse commands correctly
 */
TEST(parse_test, HandlesInput)
{
    parser p;

    char line[BUFLEN];
    char *largv[BUFLEN];

    sprintf(line, "sh doall.sh");
    p.parse(line, largv);

    const char *targv[2];
    targv[0] = "sh";
    targv[1] = "doall.sh";

    for (int i = 0; i < 2; i++)
    {
        EXPECT_STREQ(largv[i], targv[i]);
    }

    //testing space at the start of line
    sprintf(line, "   sh doall.sh");
    p.parse(line, largv);
    for (int i = 0; i < 2; i++)
    {
        EXPECT_STREQ(largv[i], targv[i]);
    }

    sprintf(line, "bash doall.sh out 2>&1");
    p.parse(line, largv);

    const char *ttargv[4];
    ttargv[0] = "bash";
    ttargv[1] = "doall.sh";
    ttargv[2] = "out";
    ttargv[3] = "2>&1";

    for (int i = 0; i < 4; i++)
    {
        EXPECT_STREQ(largv[i], ttargv[i]);
    }
};

/*! 
 * Tests parse class in translating the text file
 */
TEST(parse_cmd, HandlesCmd)
{
    parser p;

    char line[BUFLEN];
    char *largv[4];

    double lowerbound = 0;
    double upperbound = 0;

    double prior_mu1 = 0;
    double prior_mu2 = 0;

    int PopSize = 0;

    int n = 4;
    while (n--)
    {
        sprintf(line, "B%d              %d.05           10%d.012", n, n, n);
        p.parse(line, largv);

        std::string str(largv[0]);

        EXPECT_EQ(str, "B" + std::to_string(n));

        p.parse(largv[1], lowerbound);
        EXPECT_EQ(lowerbound, 0.05 + (double)n);

        p.parse(largv[2], upperbound);
        EXPECT_EQ(upperbound, 100.012 + (double)n);
    }

    n = 8;
    while (n--)
    {
        sprintf(line, " PopSize 5000%d ", n);
        p.parse(line, largv);

        std::string str(largv[0]);
        EXPECT_EQ(str, "PopSize");

        p.parse(largv[1], PopSize);
        EXPECT_EQ(PopSize, 50000 + n);
    }

    sprintf(line, "   prior_mu    1.0,0.1 ");
    p.parse(line, largv);

    std::string str(largv[0]);

    p.parse(largv[1], prior_mu1);
    EXPECT_EQ(prior_mu1, 1.0);

    p.parse(largv[2], prior_mu2);
    EXPECT_EQ(prior_mu2, 0.1);
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}