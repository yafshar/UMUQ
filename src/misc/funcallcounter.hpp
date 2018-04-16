#ifndef UMHBM_FUNCALLCOUNTER_H
#define UMHBM_FUNCALLCOUNTER_H

/*! \class funcallcounter
* 
* \brief Function call counters
*	
*/
class funcallcounter
{
  public:
    static pthread_mutex_t function_counter_mutex;

    static int num_of_local_function_counter;
    static int num_of_global_function_counter;
    static int num_of_total_function_counter;

    /*! 
     * \brief Registers tasks
     */
    bool init()
    {
        auto initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            return false;
        }

        torc_register_task((void *)reset_Task);
        torc_register_task((void *)count_Task);

        return true;
    }

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
    void reset();

    /*! 
     * \brief task of getting the local function call counters
     */
    static void count_Task(int *x);

    /*! 
     * \returns count the Global number of function calls
     */
    void count();

    /*! 
     * \returns the Total number of function calls
     */
    int get_nlocalfc() { return funcallcounter::num_of_local_function_counter; }

    /*! 
     * \returns the Total number of function calls
     */
    int get_nglobalfc() { return funcallcounter::num_of_global_function_counter; }

    /*! 
     * \returns the Total number of function calls
     */
    int get_ntotalfc() { return funcallcounter::num_of_total_function_counter; }
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
 * \brief Resetting the local function call counters to zero
 */
void funcallcounter::reset()
{
    for (int i = 0; i < torc_num_nodes(); i++)
    {
        torc_create_ex(i * torc_i_num_workers(), 1, (void (*)())funcallcounter::reset_Task, 0);
    }
    torc_waitall();
}

/*! 
 * \brief Get task of the local function call counters
 */
void funcallcounter::count_Task(int *x) { *x = funcallcounter::num_of_local_function_counter; }

void funcallcounter::count()
{
    //TODO correct the number of nodes
    int c[1024]; /* MAX_NODES*/

    for (int i = 0; i < torc_num_nodes(); i++)
    {
        torc_create_ex(i * torc_i_num_workers(), 1, (void (*)())funcallcounter::count_Task, 1,
                       1, MPI_INT, CALL_BY_RES,
                       &c[i]);
    }
    torc_waitall();

    funcallcounter::num_of_global_function_counter = std::accumulate(c, c + torc_num_nodes(), 0);

    funcallcounter::num_of_total_function_counter += funcallcounter::num_of_global_function_counter;
}

#endif
