#ifndef UMUQ_FUNCALLCOUNTER_H
#define UMUQ_FUNCALLCOUNTER_H

//! Mutex object
static std::mutex function_counter_m;

//! Local number of function calls
static int num_of_local_function_counter = 0;

//! Global number of function calls
static int num_of_global_function_counter = 0;

//! Total number of function calls
static int num_of_total_function_counter = 0;

//! True if Tasks have been registered, and false otherwise (logical).
static bool funcallcounter_Task_registered = false;

/*! \class funcallcounter
* 
* \brief Function call counters
*	
*/
class funcallcounter
{
  public:
    /*!
     * \brief Construct a new funcallcounter object
     * 
     */
    funcallcounter();

    /*!
     * \brief Destroy the funcallcounter object
     * 
     */
    ~funcallcounter();

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
    inline int getLocalFunctionCallsNumber() { return num_of_local_function_counter; }

    /*!
     * \brief Get the number of global function calls
     * 
     * \return The number of global function calls 
     */
    inline int getGlobalFunctionCallsNumber() { return num_of_global_function_counter; }

    /*!
     * \brief Get the total number of function calls
     * 
     * \return The total number of function calls
     */
    inline int getTotalFunctionCallsNumber() { return num_of_total_function_counter; }
};

funcallcounter::funcallcounter()
{
    std::lock_guard<std::mutex> lock(function_counter_m);

    // Check if psrandom is already initilized
    if (!funcallcounter_Task_registered)
    {
        torc_register_task((void *)funcallcounter::reset_Task);
        torc_register_task((void *)funcallcounter::count_Task);

        funcallcounter_Task_registered = true;
    }
}

funcallcounter::~funcallcounter() {}

/*! 
 * \brief Initize and registers tasks on TORC library
 * 
 * NOTE: 
 * init should be called before calling any other functions
 */
inline bool funcallcounter::init()
{
    // Make sure MPI is initilized
    auto initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        UMUQFAILRETURN("Failed to initilize MPI!");
    }

    return true;
}

/*! 
 * \brief Increment the local function call counters
 * 
 */
inline void funcallcounter::increment()
{
    std::lock_guard<std::mutex> lock(function_counter_m);
    num_of_local_function_counter++;
}

/*! 
 * \brief Resetting the local function call counters to zero
 * 
 */
inline void funcallcounter::reset_Task() { num_of_local_function_counter = 0; }

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
inline void funcallcounter::count_Task(int *x) { *x = num_of_local_function_counter; }

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

    num_of_global_function_counter = std::accumulate(c.begin(), c.end(), 0);
    num_of_total_function_counter += num_of_global_function_counter;
}

#endif
