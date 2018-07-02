#include "core/core.hpp"
#include "core/environment.hpp"
#include "misc/utility.hpp"
#include "misc/parser.hpp"
#include "gtest/gtest.h"

// Tests
TEST(execute_cmd_test, HandlesZeroInput)
{
    parser p;
    utility u;
    char *argv[LINESIZE];
    p.parse("mkdir -p testUtility");
    std::size_t i = 0;
    p.getLineArg(argv, i);
    EXPECT_TRUE(i == 3);
    EXPECT_TRUE(u.execute_cmd(torc_node_id(), argv, nullptr));

    p.parse("rm -fr testUtility");
    p.getLineArg(argv, i);
    EXPECT_TRUE(i == 3);
    EXPECT_TRUE(u.execute_cmd(torc_node_id(), argv, nullptr));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}