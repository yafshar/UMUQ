#include "core/core.hpp"
#include "environment.hpp"
#include "misc/utility.hpp"
#include "misc/parser.hpp"
#include "gtest/gtest.h"

// Test ExecuteCommand utility function
TEST(utility_test, HandlesExecuteCommand)
{
    // Create an instance of parser
    parser p;
    // Create an instance of utility object
    utility u;

    char *argv[LINESIZE];
    std::size_t ArgNum;

    // Create a new directory
    p.parse("mkdir testUtility");
    p.getLineArg(argv, ArgNum);

    EXPECT_TRUE(ArgNum == 2);
    EXPECT_TRUE(u.executeCommand(torc_node_id(), argv));

    p.parse("test -d testUtility");
    p.getLineArg(argv, ArgNum);

    EXPECT_TRUE(ArgNum == 3);
    EXPECT_TRUE(u.executeCommand(torc_node_id(), argv));

    p.parse("rm -fr testUtility");
    p.getLineArg(argv, ArgNum);

    EXPECT_TRUE(ArgNum == 3);
    EXPECT_TRUE(u.executeCommand(torc_node_id(), argv));

    EXPECT_TRUE(u.executeCommand("!(test -d testUtility)"));
    EXPECT_TRUE(u.executeCommand("mkdir testUtility"));
    EXPECT_TRUE(u.executeCommand("test -d testUtility"));
    EXPECT_TRUE(u.executeCommand("rm -fr testUtility"));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment<>);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new UMUQEventListener);

    return RUN_ALL_TESTS();
}