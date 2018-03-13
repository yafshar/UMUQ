#include <iostream>
#include <stdio.h>

#include "../src/misc/spawner.hpp"
#include "gtest/gtest.h"

#define BUFLEN 1024

// Tests parse
TEST(parse_test, HandlesZeroInput)
{
    spawner s;

    char line[BUFLEN];
    char *largv[BUFLEN];

    sprintf(line, " ");
    s.parse(line, largv);

    const char *targv[0];
    targv[0] = "";

    EXPECT_STREQ(largv[0], targv[0]);

    //checking for tab and next line
    sprintf(line, "\t  \n");

    s.parse(line, largv);
    EXPECT_STREQ(largv[0], targv[0]);
};

TEST(parse_test, HandlesInput)
{
    spawner s;

    char line[BUFLEN];
    char *largv[BUFLEN];

    sprintf(line, "sh doall.sh");
    s.parse(line, largv);

    const char *targv[2];
    targv[0] = "sh";
    targv[1] = "doall.sh";

    for (int i = 0; i < 2; i++)
    {
        EXPECT_STREQ(largv[i], targv[i]);
    }

    //testing space at the start of line
    sprintf(line, "   sh doall.sh");
    s.parse(line, largv);
    for (int i = 0; i < 2; i++)
    {
        EXPECT_STREQ(largv[i], targv[i]);
    }

    sprintf(line, "bash doall.sh out 2>&1");
    s.parse(line, largv);

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

TEST(fileExist, HandlesFiles)
{
    spawner s;
    
    //EXPECT_EXIT(s.fileExists("TMCMC.par"), ::testing::KilledBySignal(1), "");
};

TEST(parse_cmd, HandlesCmd)
{
    spawner s;
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}