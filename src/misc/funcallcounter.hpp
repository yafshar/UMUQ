#ifndef UMUQ_FUNCALLCOUNTER_H
#define UMUQ_FUNCALLCOUNTER_H

/*! \class funcallcounter
* 
* \brief Function call counters
*	
*/
class funcallcounter
{
  public:
    /*! 
     * \brief Initize and registers tasks on TORC library
     * 
     * NOTE: 
     * init should be called before calling any other functions
     */
    inline bool init();

    /*! 
     * \brief Increment the local function call counters
     * 
     */
    inline void increment();

    /*! 
     * \brief Reset task of setting the local function call counters to zero
     * 
     */
    static inline void reset_Task();

    /*! 
     * \brief Resetting the local function call counters to zero
     * 
     */
    inline void reset();

    /*! 
     * \brief Task of getting the local function call counters
     * 
     */
    static inline void count_Task(int *x);

    /*! 
     * \returns Count the Global number of function calls
     * 
     */
    inline void count();

    /*!
     * \brief Get the number of local function calls
     * 
     * \returns The number of local function calls
     */
    inline int getLocalFunctionCallsNumber() { return funcallcounter::num_of_local_function_counter; }

    /*!
     * \brief Get the number of global function calls
     * 
     * \return The number of global function calls 
     */
    inline int getGlobalFunctionCallsNumber() { return funcallcounter::num_of_global_function_counter; }

    /*!
     * \brief Get the total number of function calls
     * 
     * \return The total number of function calls
     */
    inline int getTotalFunctionCallsNumber() { return funcallcounter::num_of_total_function_counter; }

  public:
    //! Mutex object
    static std::mutex function_counter_mutex;

    //! Local number of function calls
    static int num_of_local_function_counter;

    //! Global number of function calls
    static int num_of_global_function_counter;

    //! Total number of function calls
    static int num_of_total_function_counter;
};

// Initialization of the static members outside of the class declaration
//! Mutex object
std::mutex funcallcounter::function_counter_mutex;
//! Local number of function calls
int funcallcounter::num_of_local_function_counter = 0;
//! Global number of function calls
int funcallcounter::num_of_global_function_counter = 0;
//! Total number of function calls
int funcallcounter::num_of_total_function_counter = 0;

/*! 
 * \brief Initize and registers tasks on TORC library
 * 
 * NOTE: 
 * init should be called before calling any other functions
 */
inline bool funcallcounter::init()
{
    auto initialized(0);
    MPI_Initialized(&initialized);
    if (initialized)
    {
        torc_register_task((void *)funcallcounter::reset_Task);
        torc_register_task((void *)funcallcounter::count_Task);

        return true;
    }
    UMUQFAILRETURN("MPI is not initialized! \n You should Initialize torc first!");
}

/*! 
 * \brief Increment the local function call counters
 * 
 */
inline void funcallcounter::increment()
{
    std::lock_guard<std::mutex> lock(funcallcounter::function_counter_mutex);
    funcallcounter::num_of_local_function_counter++;
}

/*! 
 * \brief Resetting the local function call counters to zero
 * 
 */
inline void funcallcounter::reset_Task() { funcallcounter::num_of_local_function_counter = 0; }

/*! 
 * \brief Resetting the local function call counters to zero
 * 
 */
inline void funcallcounter::reset()
{
    for (int i = 0; i < torc_num_nodes(); i++)
    {
        torc_create_ex(i * torc_i_num_workers(), 1, (void (*)())funcallcounter::reset_Task, 0);
    }
    torc_waitall();
}

/*! 
 * \brief Task of getting the local function call counters
 * 
 */
inline void funcallcounter::count_Task(int *x) { *x = funcallcounter::num_of_local_function_counter; }

/*!
 * \returns Count the Global number of function calls
 * 
 */
inline void funcallcounter::count()
{
    int const maxNumNodes = torc_num_nodes();
    std::vector<int> c(maxNumNodes, 0);
    for (int i = 0; i < maxNumNodes; i++)
    {
        int *cp = c.data() + i;
        torc_create_ex(i * torc_i_num_workers(), 1, (void (*)())funcallcounter::count_Task, 1,
                       1, MPI_INT, CALL_BY_REF,
                       cp);
    }
    torc_waitall();

    funcallcounter::num_of_global_function_counter = std::accumulate(c.begin(), c.end(), 0);
    funcallcounter::num_of_total_function_counter += funcallcounter::num_of_global_function_counter;
}

#endif
