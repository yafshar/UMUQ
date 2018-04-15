#ifndef UMHBM_FUNCALLCOUNTER_H
#define UMHBM_FUNCALLCOUNTER_H

/*! \class funcallcounter
* 
* \brief Function call counters
*	
*/
struct funcallcounter
{
	static pthread_mutex_t function_counter_mutex;

	static int num_of_local_function_counter;
	static int num_of_global_function_counter;
	static int num_of_total_function_counter;

	/*! 
     * \brief Increment the local function call counters
     */
	void increment()
	{
		//lock and unlock a mutex
		pthread_mutex_lock(&function_counter_mutex);
		num_of_local_function_counter++;
		pthread_mutex_unlock(&function_counter_mutex);
	}

	/*! 
     * \brief Reset task of setting the local function call counters to zero
     */
	static void reset_Task();

	/*! 
     * \brief Resetting the local function call counters to zero
     */
	void reset()
	{
		for (int i = 0; i < torc_num_nodes(); i++)
		{
			torc_create_ex(i * torc_i_num_workers(), 1, (void *)reset_Task, 0);
		}
		torc_waitall();
	}

	/*! 
     * \brief Get task of the local function call counters
     */
	static void get_Task(int *x);

	int get()
	{
		//TODO correct the number of nodes
		int c[1024]; /* MAX_NODES*/

		for (int i = 0; i < torc_num_nodes(); i++)
		{
			torc_create_ex(i * torc_i_num_workers(), 1, (void *)get_Task, 1,
						   1, MPI_INT, CALL_BY_RES,
						   &c[i]);
		}
		torc_waitall();

		num_of_global_function_counter = std::accumulate(c, c + torc_num_nodes(), 0);

		std::cout << "global number of function counts: " << num_of_global_function_counter << std::endl;

		num_of_total_function_counter += num_of_global_function_counter;

		return num_of_global_function_counter;
	}
};

pthread_mutex_t funcallcounter::function_counter_mutex = PTHREAD_MUTEX_INITIALIZER;

int funcallcounter::num_of_local_function_counter = 0;
int funcallcounter::num_of_global_function_counter = 0;
int funcallcounter::num_of_total_function_counter = 0;

/*! 
 * \brief Resetting the local function call counters to zero
 */
void funcallcounter::reset_Task() { funcallcounter::num_of_local_function_counter = 0; }

/*! 
 * \brief Get task of the local function call counters
 */
void funcallcounter::get_Task(int *x) { *x = funcallcounter::num_of_local_function_counter; }

#endif
