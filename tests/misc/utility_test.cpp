#include "core/core.hpp"
#include "environment.hpp"
#include "misc/utility.hpp"
#include "gtest/gtest.h"

// Test ExecuteCommand utility function
TEST(utility_test, HandlesExecuteCommand)
{
    // Create an instance of utility object
    umuq::utility u;

    EXPECT_TRUE(u.executeCommand(torc_node_id(), "mkdir testUtility"));
    EXPECT_TRUE(u.executeCommand(torc_node_id(), "rm -fr testUtility"));

    std::vector<std::string> commands;

    {
        std::string command("mkdir testUtility");
        commands.push_back(command);
    }

    {
        std::string command("ls -l > out 2>&1 &");
        commands.push_back(command);
    }

    {
        std::string command("mv out testUtility");
        commands.push_back(command);
    }
    EXPECT_TRUE(u.executeCommands(torc_node_id(), commands, "testUtility", true));

    EXPECT_TRUE(u.changeWorkingDirectory("./testUtility"));
    EXPECT_TRUE(u.executeCommand("cp out out2"));
    EXPECT_TRUE(u.changeWorkingDirectory(".."));
    EXPECT_TRUE(u.executeCommand("rm -fr testUtility"));
}

/* Threads can modify \c sharedId, mutex will protect this variable */
int sharedId = 0;

/* Only one thread can modify \c exclusiveId no protection needed */
int exclusiveId = 0;

/*!
 * \brief Utility task, where main thread can modify both \c sharedId and \c exclusiveId
 *
 */
void utilityTask1()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Create an instance of utility object
    umuq::utility u;

    for (;;)
    {
        if (u.tryLockOrSleep(100, 1))
        {
            std::cout << "shared Id =" << sharedId << std::endl;
            umuq::utility_m.unlock();
            return;
        }
        else
        {
            // Can't get lock to modify sharedId, but there is some other work to do
            exclusiveId++;
            std::cout << "exclusive Id =" << exclusiveId << std::endl;
        }
    }
}

/*!
 * \brief Utility task, where this thread can modify only \c sharedId
 *
 */
void utilityTask2()
{
    std::lock_guard<std::mutex> lock(umuq::utility_m);
    sharedId++;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

// Test tryLockOrSleep utility function
TEST(utility_test, HandlesTryLockOrSleep)
{
    torc_register_task((void *)utilityTask1);
    torc_register_task((void *)utilityTask2);
    torc_create(-1, (void (*)())utilityTask2, 0);
    torc_create(-1, (void (*)())utilityTask1, 0);
    torc_waitall();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}