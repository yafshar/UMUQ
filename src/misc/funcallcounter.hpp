#ifndef UMUQ_FUNCALLCOUNTER_H
#define UMUQ_FUNCALLCOUNTER_H

namespace umuq
{
//! Local number of function calls
static int nLocalFunctionCounter = 0;

//! Global number of function calls
static int nGlobalFunctionCounter = 0;

//! Total number of function calls
static int nTotalFunctionCounter = 0;

//! True if Tasks have been registered, and false otherwise (logical).
static bool isFuncallcounterTaskRegistered = false;

//! Mutex object
static std::mutex functionCounter_m;

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
     * \brief Initializes and registers tasks on TORC library
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
    static inline void resetTask();

    /*! 
     * \brief Resetting the local function call counters to zero
     * 
     */
    inline void reset();

    /*! 
     * \brief Task of getting the local function call counters
     * 
     */
    static inline void countTask(int *x);

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
    inline int getLocalFunctionCallsNumber() { return nLocalFunctionCounter; }

    /*!
     * \brief Get the number of global function calls
     * 
     * \return The number of global function calls 
     */
    inline int getGlobalFunctionCallsNumber() { return nGlobalFunctionCounter; }

    /*!
     * \brief Get the total number of function calls
     * 
     * \return The total number of function calls
     */
    inline int getTotalFunctionCallsNumber() { return nTotalFunctionCounter; }
};

funcallcounter::funcallcounter()
{
    std::lock_guard<std::mutex> lock(functionCounter_m);

    // Check if psrandom is already initialized
    if (!isFuncallcounterTaskRegistered)
    {
        torc_register_task((void *)funcallcounter::resetTask);
        torc_register_task((void *)funcallcounter::countTask);

        isFuncallcounterTaskRegistered = true;
    }
}

funcallcounter::~funcallcounter() {}

/*! 
 * \brief Initialize and registers tasks on TORC library
 * 
 * NOTE: 
 * init should be called before calling any other functions
 */
inline bool funcallcounter::init()
{
    // Make sure MPI is initialized
    auto initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        UMUQFAILRETURN("Failed to initialize MPI!");
    }

    return true;
}

/*! 
 * \brief Increment the local function call counters
 * 
 */
inline void funcallcounter::increment()
{
    std::lock_guard<std::mutex> lock(functionCounter_m);
    nLocalFunctionCounter++;
}

/*! 
 * \brief Resetting the local function call counters to zero
 * 
 */
inline void funcallcounter::resetTask() { nLocalFunctionCounter = 0; }

/*! 
 * \brief Resetting the local function call counters to zero
 * 
 */
inline void funcallcounter::reset()
{
    for (int i = 0; i < torc_num_nodes(); i++)
    {
        torc_create_ex(i * torc_i_num_workers(), 1, (void (*)())funcallcounter::resetTask, 0);
    }
    torc_waitall();
}

/*! 
 * \brief Task of getting the local function call counters
 * 
 */
inline void funcallcounter::countTask(int *x) { *x = nLocalFunctionCounter; }

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
        torc_create_ex(i * torc_i_num_workers(), 1, (void (*)())funcallcounter::countTask, 1,
                       1, MPI_INT, CALL_BY_REF,
                       cp);
    }
    torc_waitall();

    nGlobalFunctionCounter = std::accumulate(c.begin(), c.end(), 0);
    nTotalFunctionCounter += nGlobalFunctionCounter;
}

} // namespace umuq

#endif // UMUQ_FUNCALLCOUNTER
