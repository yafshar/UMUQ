#ifndef UMUQ_ENVIRONMENT_H
#define UMUQ_ENVIRONMENT_H

#include "core/core.hpp"
#include "misc/funcallcounter.hpp"
#include "data/database.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"

#if HAVE_GOOGLETEST == 1 && defined(UMUQ_UNITTEST)
#include "gtest/gtest.h"
#endif // HAVE_GOOGLETEST

namespace umuq
{

/*!
 * \brief torcEnvironment class
 * 
 */
template <typename T = double>
class torcEnvironment
#if HAVE_GOOGLETEST == 1 && defined(UMUQ_UNITTEST)
	: public ::testing::Environment
#endif
{
  public:
	/*!
     * \brief Construct a new torc Environment object
     *
     */
	torcEnvironment()
#if HAVE_GOOGLETEST == 1 && defined(UMUQ_UNITTEST)
		: ::testing::Environment()
#endif
	{
	}
	
	/*!
     * \brief This would register the function in TORC library
     * 
     * \param f function casted to void *
     */
	template <class F>
	void register_task(F f)
	{
		if (f)
		{
			torc_register_task((void *)f);
		}
	}

	/*!
     * \brief Set up the torc object
     * 
     */
	virtual void SetUp()
	{
		if (!PRNG_Task_registered<T>)
		{
			torc_register_task((void *)psrandom<T>::init_Task);

			PRNG_Task_registered<T> = true;
		}

		if (!funcallcounter_Task_registered)
		{
			torc_register_task((void *)funcallcounter::reset_Task);
			torc_register_task((void *)funcallcounter::count_Task);

			funcallcounter_Task_registered = true;
		}

		if (!umuq::tmcmc::database_update_Task_registered<T>)
		{
			torc_register_task((void *)umuq::tmcmc::update_Task<T>);

			umuq::tmcmc::database_update_Task_registered<T> = true;
		}

		char **argv = NULL;
		int argc = 0;

		torc_init(argc, argv);
	}

	/*!
     * \brief Set up the torc object
     * 
     * \param argc 
     * \param argv 
     */
	virtual void SetUp(int argc, char **argv)
	{
		if (!PRNG_Task_registered<T>)
		{
			torc_register_task((void *)psrandom<T>::init_Task);
			PRNG_Task_registered<T> = true;
		}

		if (!funcallcounter_Task_registered)
		{
			torc_register_task((void *)funcallcounter::reset_Task);
			torc_register_task((void *)funcallcounter::count_Task);

			funcallcounter_Task_registered = true;
		}

		if (umuq::tmcmc::database_update_Task_registered<T>)
		{
			torc_register_task((void *)umuq::tmcmc::update_Task<T>);

			umuq::tmcmc::database_update_Task_registered<T> = true;
		}

		torc_init(argc, argv);
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
	torcEnvironment(torcEnvironment<T> const &) = delete;

	// Make it not assignable
	torcEnvironment<T> &operator=(torcEnvironment<T> const &) = delete;
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

#endif // HAVE_GOOGLETEST == 1

} // namespace umuq

#endif // UMUQ_ENVIRONMENT
