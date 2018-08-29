#ifndef UMUQ_ENVIRONMENT_H
#define UMUQ_ENVIRONMENT_H

#ifdef HAVE_CONFIG_H
#include <UMUQ_config.h>
#endif /* HAVE_CONFIG_H */

// #define HAVE_TORC 1
// #define HAVE_GOOGLETEST 1
// #define UMUQ_UNITTEST

#if HAVE_TORC == 1
#include <torc.h>
#if HAVE_GOOGLETEST == 1 && defined(UMUQ_UNITTEST)
#include "gtest/gtest.h"

/*!
 * \brief torcEnvironment class
 * 
 */
class torcEnvironment : public ::testing::Environment
{
  public:
    /*!
     * \brief Construct a new torc Environment object
     *
     */
    torcEnvironment() : ::testing::Environment() {}
#else
class torcEnvironment
{
  public:
    torcEnvironment(){};
#endif //HAVE_GOOGLETEST

    /*!
     * \brief Set up the torc object
     * 
     */
    virtual void SetUp()
    {
        char **argv = NULL;
        int argc = 0;

        torc_init(argc, argv, 0);
    }

    /*!
     * \brief Set up the torc object
     * 
     * \param argc 
     * \param argv 
     */
    virtual void SetUp(int argc, char **argv)
    {
        torc_init(argc, argv, 0);
    }

    /*!
     * \brief Finish the torc
     * 
     */
    virtual void TearDown()
    {
        torc_finalize();
    }

    /*!
     * \brief Destroy the torc Environment object
     * 
     */
    virtual ~torcEnvironment() {}

  private:
    // Make it noncopyable
    torcEnvironment(torcEnvironment const &) = delete;

    // Make it not assignable
    torcEnvironment &operator=(torcEnvironment const &) = delete;
};

#if HAVE_GOOGLETEST == 1 && defined(UMUQ_UNITTEST)
/*!
 * \brief New event listener to make sure of Abort in case of failure
 * 
 */
class UMUQEventListener : public ::testing::EmptyTestEventListener
{
  public:
    UMUQEventListener() : ::testing::EmptyTestEventListener(),
                          result_vector() {}

    ~UMUQEventListener() {}

    // Called before a test starts.
    virtual void OnTestStart(::testing::TestInfo const &test_info)
    {
    }

    // Called after an assertion failure or an explicit SUCCESS() macro.
    virtual void OnTestPartResult(::testing::TestPartResult const &test_part_result)
    {
        result_vector.push_back(test_part_result);
    }

    // Called after a test ends.
    virtual void OnTestEnd(::testing::TestInfo const &test_info)
    {
        for (std::size_t i = 0; i < result_vector.size(); i++)
        {
            ::testing::TestPartResult const test_part_result = result_vector.at(i);
            if (test_part_result.failed())
            {
                std::string UMUQ_message = "Error: " + std::string(test_part_result.file_name()) + ":" + std::to_string(test_part_result.line_number());
                UMUQFAIL(UMUQ_message);
            }
        }
        result_vector.clear();
    }

  private:
    std::vector<::testing::TestPartResult> result_vector;
};

#endif
#else
#define CALL_BY_COP (int)(0x0001)
#define CALL_BY_REF (int)(0x0002)
#define CALL_BY_RES (int)(0x0003)
#define CALL_BY_PTR (int)(0x0004)
#define CALL_BY_VAL (int)(0x0001)
#define CALL_BY_COP2 (int)(0x0005)
#define CALL_BY_VAD (int)(0x0006)

void torc_init(int argc, char *argv[], int ms);
void torc_reset_statistics();

int torc_i_worker_id(void);
int torc_i_num_workers();
int torc_worker_id();
int torc_num_workers();
int torc_getlevel();

void torc_enable_stealing();
void torc_disable_stealing();
void torc_i_enable_stealing();
void torc_i_disable_stealing();
void start_server_thread();
void shutdown_server_thread();
void torc_taskinit();
void torc_waitall();
void torc_waitall2();
void torc_waitall3();
void torc_tasksync();

int torc_scheduler_loop(int);

void torc_task(int queue, void (*f)(), int narg, ...);
void torc_task_detached(int queue, void (*f)(), int narg, ...);
void torc_task_ex(int queue, int invisible, void (*f)(), int narg, ...);
void torc_task_direct(int queue, void (*f)(), int narg, ...);

#define torc_create torc_task
#define torc_create_detached torc_task_detached
#define torc_create_ex torc_task_ex
#define torc_create_direct torc_task_direct

int torc_node_id();
int torc_num_nodes();
void torc_broadcast(void *a, long count, MPI_Datatype dtype);
void torc_broadcast_ox(void *a, long count, int dtype);
void thread_sleep(int ms);
void torc_finalize(void);
void torc_register_task(void *f);
int torc_fetch_work();

/*!
 * \brief torcEnvironment class
 *
 */
class torcEnvironment
{
  public:
    /*!
     * \brief Set up the torc object
     *
     */
    virtual void SetUp() {}

    /*!
     * \brief Set up the torc object
     *
     * \param argc
     * \param argv
     */
    virtual void SetUp(int argc, char **argv) {}

    /*!
     * \brief Finish the torc
     *
     */
    virtual void TearDown() {}

    /*!
     * \brief Destroy the torc Environment object
     *
     */
    virtual ~torcEnvironment() {}
};
#endif //HAVE_TORC
#endif //UMUQ_ENVIRONMENT_H
