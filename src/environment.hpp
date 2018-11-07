#ifndef UMUQ_ENVIRONMENT_H
#define UMUQ_ENVIRONMENT_H

#include "core/core.hpp"
#include "misc/funcallcounter.hpp"
#include "data/database.hpp"
#include "data/runinfo.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"

#if HAVE_GOOGLETEST == 1 && defined(UMUQ_UNITTEST)
#include "gtest/gtest.h"
#endif // HAVE_GOOGLETEST

namespace umuq
{

/*! \class torcEnvironment
 * \ingroup Core_Module
 * 
 * \brief Create a new torc Environment object.
 * 
 * An Environment object is capable of setting up and tearing down an
 * environment. It does the set-up and tear-down in virtual methods 
 * SetUp() and TearDown() instead of the constructor and the destructor.
 * 
 * \tparam T Data type (default is double) 
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
		if (!isPrngTaskRegistered<T>)
		{
			torc_register_task((void *)psrandom<T>::initTask);

			isPrngTaskRegistered<T> = true;
		}

		if (!isFuncallcounterTaskRegistered)
		{
			torc_register_task((void *)funcallcounter::resetTask);
			torc_register_task((void *)funcallcounter::countTask);

			isFuncallcounterTaskRegistered = true;
		}

		if (!umuq::tmcmc::isUpdateTaskRegistered<T>)
		{
			torc_register_task((void *)umuq::tmcmc::updateDataTask<T>);

			umuq::tmcmc::isUpdateTaskRegistered<T> = true;
		}

		if (!umuq::tmcmc::isBroadcastTaskRegistered<T>)
		{
			torc_register_task((void *)umuq::tmcmc::broadcastTask<T>);

			umuq::tmcmc::isBroadcastTaskRegistered<T> = true;
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
		if (!isPrngTaskRegistered<T>)
		{
			torc_register_task((void *)psrandom<T>::initTask);
			isPrngTaskRegistered<T> = true;
		}

		if (!isFuncallcounterTaskRegistered)
		{
			torc_register_task((void *)funcallcounter::resetTask);
			torc_register_task((void *)funcallcounter::countTask);

			isFuncallcounterTaskRegistered = true;
		}

		if (umuq::tmcmc::isUpdateTaskRegistered<T>)
		{
			torc_register_task((void *)umuq::tmcmc::updateDataTask<T>);
			umuq::tmcmc::isUpdateTaskRegistered<T> = true;
		}

		if (!umuq::tmcmc::isBroadcastTaskRegistered<T>)
		{
			torc_register_task((void *)umuq::tmcmc::broadcastTask<T>);

			umuq::tmcmc::isBroadcastTaskRegistered<T> = true;
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

  protected:
	/*!
     * \brief Delete a torcEnvironment object copy construction
     * 
     * Make it noncopyable.
     */
	torcEnvironment(torcEnvironment<T> const &) = delete;

	/*!
     * \brief Delete a torcEnvironment object assignment
     * 
     * Make it nonassignable
     * 
     * \returns torcEnvironment<T>& 
     */
	torcEnvironment<T> &operator=(torcEnvironment<T> const &) = delete;

#if HAVE_MPI == 1
	/**
	 * \todo
	 * Complete the communicator so the algorithm can work on different groups
	 */
  private:
	//! Communicator for MPI
	MPI_Comm comm;
#endif
};

#if HAVE_GOOGLETEST == 1 && defined(UMUQ_UNITTEST)
/*! \class UMUQEventListener
 * \ingroup Test_Module
 * 
 * \brief New event listener to make sure of Abort in case of failure
 * 
 * An interface to %UMUQ for tracing execution of tests from [Google Test](https://github.com/google/googletest).<br>
 * The methods are organized in the order the corresponding events are fired.
 */
class UMUQEventListener : public ::testing::EmptyTestEventListener
{
  public:
	/*!
     * \brief Construct a new UMUQEventListener object
     * 
     */
	UMUQEventListener() : ::testing::EmptyTestEventListener(),
						  result_vector() {}

	/*!
	 * \brief Destroy the UMUQEventListener object
	 * 
	 */
	~UMUQEventListener() {}

	/*!
	 * \brief It is called before a test starts.
	 * 
	 * \param test_info 
	 */
	virtual void OnTestStart(::testing::TestInfo const &test_info)
	{
	}

	/*!
	 * \brief Called after an assertion failure or an explicit \c SUCCESS() macro.
	 * 
	 * \param test_part_result 
	 */
	virtual void OnTestPartResult(::testing::TestPartResult const &test_part_result)
	{
		result_vector.push_back(test_part_result);
	}

	/*!
	 * \brief Called after a test ends.
	 * 
	 * \param test_info 
	 */
	virtual void OnTestEnd(::testing::TestInfo const &test_info)
	{
		for (std::size_t i = 0; i < result_vector.size(); i++)
		{
			::testing::TestPartResult const test_part_result = result_vector.at(i);
			if (test_part_result.failed())
			{
				UMUQFAIL(test_part_result.file_name(), ":", test_part_result.line_number());
			}
		}
		result_vector.clear();
	}

  private:
	std::vector<::testing::TestPartResult> result_vector;
};

#endif // HAVE_GOOGLETEST == 1

/*!
 * \ingroup Core_Module
 * 
 * \brief TORC environemnt object
 * 
 * \tparam T Data type
 */
template <typename T = double>
std::unique_ptr<torcEnvironment<T>> Torc;

} // namespace umuq

#endif // UMUQ_ENVIRONMENT
