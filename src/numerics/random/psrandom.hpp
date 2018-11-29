#ifndef UMUQ_PSRANDOM_H
#define UMUQ_PSRANDOM_H

#include "core/core.hpp"
#include "numerics/factorial.hpp"

/*! 
 * \defgroup Random_Module Random distribution module
 * \ingroup Numerics_Module
 * 
 * This is the random module of %UMUQ providing all necessary classes that generate random number distributions. <br>
 * A random number distribution post-processes the output of a uniform random generators in such a way that 
 * resulting output is distributed according to a defined statistical probability density function.
 */

#include "saruprng.hpp"

namespace umuq
{

/*!
 * \ingroup Random_Module
 * 
 * \brief pseudo-random number seed
 * 
 */
static std::size_t PRNG_seed = 0;

/*!
 * \ingroup Random_Module
 * 
 * \brief 32-bit Mersenne Twister by Matsumoto and Nishimura, 1998
 * 
 */
static std::vector<std::mt19937> NumberGenerator(1, std::move(std::mt19937(std::random_device{}())));

/*!
 * \ingroup Random_Module
 * 
 * \brief \c C++ Saru pseudo-random number generator.
 * 
 * \sa Saru.
 */
static std::vector<Saru> saru(1, std::move(Saru(std::random_device{}())));

/*!
 * \ingroup Random_Module
 * 
 * \brief It would be true if PRNG state has been initialized, and false otherwise (logical).
 * 
 */
static bool PRNG_initialized = false;

/*!
 * \ingroup Random_Module
 * 
 * \brief It would be true if Tasks have been registered, and false otherwise (logical).
 * 
 * \tparam double Data type
 */
static bool isPrngTaskRegistered = false;

/*!
 * \ingroup Random_Module
 * 
 * \brief Mutex object
 * 
 */
static std::mutex PRNG_m;

/*!
 * \ingroup Random_Module
 * 
 * \brief Uniform random of floating-point values uniformly distributed on the interval \f$ [0, 1) \f$ 
 * 
 * \returns std::uniform_real_distribution<double> Uniform random of floating-point values uniformly distributed on the interval \f$ [0, 1) \f$ 
 */
std::uniform_real_distribution<double> uniformDoubleDistribution(double{}, double{1});

/*!
 * \ingroup Random_Module
 * 
 * \brief Uniform random of floating-point values uniformly distributed on the interval \f$ [0, 1) \f$ 
 * 
 * \returns std::uniform_real_distribution<float> Uniform random of floating-point values uniformly distributed on the interval \f$ [0, 1) \f$ 
 */
std::uniform_real_distribution<float> uniformRealDistribution(float{}, float{1});

} // namespace umuq

#include "psrandom_uniformdistribution.hpp"
#include "psrandom_normaldistribution.hpp"
#include "psrandom_lognormaldistribution.hpp"
#include "psrandom_multinomialdistribution.hpp"
#include "psrandom_multivariatenormaldistribution.hpp"
#include "psrandom_exponentialdistribution.hpp"
#include "psrandom_gammadistribution.hpp"

namespace umuq
{

/*! \class psrandom
 * \ingroup Random_Module
 *
 * \brief The psrandom class includes pseudo-random numbers engines and distributions used in %UMUQ to produce pseudo-random values. 
 * 
 * This class generates pseudo-random numbers.
 * It includes engines and distributions used to produce pseudo-random values. 
 * 
 * All of the engines may be specifically seeded, for use with repeatable simulators. <br>
 * Random number engines generate pseudo-random numbers using seed data as entropy source.  <br>
 * The choice of which engine to use involves a number of tradeoffs: <br>
 * 
 * -# Saru PRNG has only a small storage requirement for state which is 64-bit and is very fast.
 * -# [The Mersenne twister](https://en.wikipedia.org/wiki/Mersenne_Twister) is slower and has greater state storage 
 *    requirements but with the right parameters has the longest non-repeating sequence with the most desirable spectral 
 *    characteristics (for a given definition of desirable). 
 * 
 * \note
 * - To use the psrandom in multithreaded application or in any class which requires setting the PRNG: <br>
 *   - First, construct a new psrandom object either with a seed or without it.
 *   - Second, initialize the PRNG or set the state of the PRNG.<br>
 *   - Third, use any member function or set the PRNG object in other classes 
 * 
 * \sa init.
 * \sa setState.
 * \sa umuq::density::densityFunction.
 */
class psrandom
{
  public:
    /*!
     * \brief Construct a new psrandom object
     * 
     */
    psrandom();

    /*!
     * \brief Construct a new psrandom object
     * 
     * \param inSeed Input seed for random number initialization 
     */
    explicit psrandom(int const inSeed);

    /*!
     * \brief Destroy the psrandom object
     * 
     */
    ~psrandom();

    /*!
     * \brief Init task on each node to set the current state of the engine for all the threads on that node 
     */
    static void initTask();

    /*!
     * \brief Set the State of psrandom object
     * 
     * \returns \c true When it successfully sets the current state of the engine. 
     */
    bool init();

    /*!
     * \brief Set the State of psrandom object
     * 
     * \returns \c true When it successfully sets the current state of the engine.
     */
    bool setState();

    /*!
     * \brief Set the PRNG seed.
     * 
     * \param inSeed  Input seed for random number initialization 
     * 
     * \returns true  When it successfully sets the seed of the engine
     * \returns false If the PRNG is already initialized or the state has been set before
     */
    inline bool setSeed(long const inSeed);

  protected:
    /*!
     * \brief Delete a psrandom object copy construction
     * 
     * Avoiding implicit generation of the copy constructor.
     */
    psrandom(psrandom const &) = delete;

    /*!
     * \brief Delete a psrandom object assignment
     * 
     * Avoiding implicit copy assignment.
     */
    psrandom &operator=(psrandom const &) = delete;

  public:
    /*!
     * \brief Uniform random of floating-point values uniformly distributed on 
     * the interval \f$ [0, 1) \f$ using a random number engine based on Mersenne Twister
     * 
     * \return double A uniform random number between \f$ [0, 1) \f$
     */
    inline double unirnd();

    /*!
     * \brief Uniform random of floating-point values uniformly distributed on 
     * the interval \f$ [low, high) \f$ using a random number engine based on Mersenne Twister
     * 
     * \param low  Lower bound of the interval (default is 0)
     * \param high  Upper bound of theinterval  (default is 1)
     * 
     * \return double A uniform random number between \f$ [low, high) \f$
     */
    inline double unirnd(double const low, double const high);

    /*!
     * \brief Uniform random of floating-point values uniformly distributed on 
     * the interval \f$ [low, high) \f$ using a random number engine based on Mersenne Twister
     * 
     * \param low  Lower bound of the interval (default is 0)
     * \param high  Upper bound of theinterval  (default is 1)
     * 
     * \return float A uniform random number between \f$ [low, high) \f$
     */
    inline float unirnd(float const low, float const high);

    /*!
     * \brief Vector of uniform random of floating-point values uniformly distributed on 
     * the interval \f$ [low, high) \f$ using a random number engine based on Mersenne Twister
     * 
     * \param idata  Array of input data of type double
     * \param nSize  Size of the array 
     * \param low    Lower bound of the interval (default is 0)
     * \param high   Upper bound of theinterval  (default is 1)
     */
    void unirnd(double *idata, int const nSize, double const low = double{}, double const high = double{1});

    /*!
     * \brief Vector of uniform random of floating-point values uniformly distributed on 
     * the interval \f$ [low, high) \f$ using a random number engine based on Mersenne Twister
     * 
     * \param idata  Array of input data of type float
     * \param nSize  Size of the array 
     * \param low    Lower bound of the interval (default is 0)
     * \param high   Upper bound of theinterval  (default is 1)
     */
    void unirnd(float *idata, int const nSize, float const low = float{}, float const high = float{1});

    /*!
     * \brief Vector of uniform random of floating-point values uniformly distributed on 
     * the interval \f$ [low, high) \f$ using a random number engine based on Mersenne Twister
     * 
     * \param idata  Array of input data of type double
     * \param low    Lower bound of the interval (default is 0)
     * \param high   Upper bound of theinterval  (default is 1)
     */
    void unirnd(std::vector<double> &idata, double const low = double{}, double const high = double{1});

    /*!
     * \brief Vector of uniform random of floating-point values uniformly distributed on 
     * the interval \f$ [low, high) \f$ using a random number engine based on Mersenne Twister
     * 
     * \param idata  Array of input data of type float
     * \param low    Lower bound of the interval (default is 0)
     * \param high   Upper bound of theinterval  (default is 1)
     */
    void unirnd(std::vector<float> &idata, float const low = float{}, float const high = float{1});

    /*!
     * \brief Vector of uniform of integer values uniformly distributed on the closed interval
     * \f$ [low, high] \f$ using a random number engine based on Mersenne Twister
     * 
     * \param idata  Array of input data of integers
     * \param nSize  Size of the array 
     * \param low    Lower bound of the interval 
     * \param high   Upper bound of theinterval
     */
    void u32rnd(int *idata, int const nSize, int const low, int const high);

    /*!
     * \brief Vector of uniform of integer values uniformly distributed on the closed interval
     * \f$ [low, high] \f$ using a random number engine based on Mersenne Twister
     * 
     * \param idata  Array of input data of integers
     * \param low    Lower bound of the interval 
     * \param high   Upper bound of theinterval
     */
    void u32rnd(std::vector<int> &idata, int const low, int const high);

    /*!
     * \brief Advance the Saru PRNG state by 1, and output a double precision \f$ [0, 1) \f$ floating point
     * 
     * \returns A uniform random number of a double precision \f$ [0, 1) \f$ floating point
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline double drnd();

    /*!
     * \brief Advance the Saru PRNG state by 1, and output a double precision \f$ [0, 1) \f$ floating point
     * 
     * \param low   Lower bound of the interval (default is 0)
     * \param high  Upper bound of theinterval  (default is 1)
     * 
     * \return double A uniform random number between \f$ [low, high) \f$
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline double drnd(double const low, double const high);

    /*!
     * \brief Advance the Saru PRNG  state by 1, and output a single precision \f$ [0, 1) \f$ floating point
     * 
     * \returns A uniform random number of a single precision \f$ [0, 1) \f$ floating point
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline float frnd();

    /*!
     * \brief Advance the Saru PRNG  state by 1, and output a single precision \f$ [0, 1) \f$ floating point
     *  
     * \param low   Lower bound of the interval (default is 0)
     * \param high  Upper bound of theinterval  (default is 1)
     * 
     * \return float A uniform random number between \f$ [low, high) \f$
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline float frnd(float const low, float const high);

    /*!
     * \brief Advance the Saru PRNG state by 1, and output a 32 bit integer pseudo-random value.
     * 
     * \returns An unsigned 32 bit integer pseudo-random value
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline unsigned int u32();

    /*!
     * \brief Advance the Saru PRNG state by 1, and output a 32 bit integer pseudo-random value 
     * with variate in \f$ [0, high] \f$ for unsigned int high
     * 
     * \returns An unsigned 32 bit integer pseudo-random value with variate in \f$ [0, high] \f$ for unsigned int high
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline unsigned int u32(unsigned int const high);

  public:
    /*!
     * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
     *
     * \tparam DataType Input data type
     * 
     * \param idata Array of input data of type DataType
     * \param nSize Size of the array idata
     *
     * \note 
     * - This should be called after setting the State of psrandom object for use 
     * in multi processes or multi threaded applications, \sa setState
     * 
     * The permutations generated by this algorithm occur with the same probability.
     *
     * Reference:<br>
     * R. Durstenfeld, "Algorithm 235: Random permutation" Communications of the ACM, 7 (1964), p. 420
     */
    template <typename DataType>
    inline void shuffle(DataType *idata, int const nSize);

    /*!
     * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
     *
     * \tparam DataType Input data type
     * 
     * \param idata Array of input data of type DataType
     * 
     * \note 
     * - This should be called after setting the State of psrandom object for use
     *  in multi processes or multi threaded applications, \sa setState
     * 
     * 
     * The permutations generated by this algorithm occur with the same probability.
     *
     * Reference:<br>
     * R. Durstenfeld, "Algorithm 235: Random permutation" Communications of the ACM, 7 (1964), p. 420
     */
    template <typename DataType>
    inline void shuffle(std::vector<DataType> &idata);
};

psrandom::psrandom()
{
    std::lock_guard<std::mutex> lock(PRNG_m);
    if (PRNG_seed == 0)
    {
        PRNG_seed = std::random_device{}();
    }
    if (!isPrngTaskRegistered)
    {
        torc_register_task((void *)psrandom::initTask);
        isPrngTaskRegistered = true;
    }
}

psrandom::psrandom(int const inSeed)
{
    std::lock_guard<std::mutex> lock(PRNG_m);
    if (PRNG_seed == 0)
    {
        PRNG_seed = static_cast<std::size_t>(inSeed);
    }
    if (!isPrngTaskRegistered)
    {
        torc_register_task((void *)psrandom::initTask);
        isPrngTaskRegistered = true;
    }
}

psrandom::~psrandom() {}

void psrandom::initTask()
{
    std::vector<std::size_t> rSeed(std::mt19937::state_size);

    // Get the local number of workers
    std::size_t const nlocalworkers = static_cast<std::size_t>(torc_i_num_workers());

    // Node Id (MPI rank)
    std::size_t const node_id = static_cast<std::size_t>(torc_node_id());

    std::size_t const n = nlocalworkers * (node_id + 1);

    for (std::size_t i = 0; i < nlocalworkers; i++)
    {
        std::size_t const j = PRNG_seed + n + i;
        std::iota(rSeed.begin(), rSeed.end(), j);

        // Seed the engine with unsigned ints
        std::seed_seq sSeq(rSeed.begin(), rSeed.end());

        // For each thread feed the RNG
        NumberGenerator[i].seed(sSeq);

        Saru s(PRNG_seed, n, i);
        saru[i] = std::move(s);
    }
}

bool psrandom::init()
{
    {
        // Make sure MPI is initialized
        auto initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            UMUQFAILRETURN("Failed to initialize MPI!");
        }
    }

    {
        std::lock_guard<std::mutex> lock(PRNG_m);

        // Check if psrandom is already initialized
        if (PRNG_initialized)
        {
            return true;
        }

        PRNG_initialized = true;
    }

    int const nlocalworkers = torc_i_num_workers();

    try
    {
        NumberGenerator.resize(nlocalworkers);
        saru.resize(nlocalworkers);
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    };

    for (int i = 0; i < torc_num_nodes(); i++)
    {
        torc_create_ex(i * nlocalworkers, 1, (void (*)())psrandom::initTask, 0);
    }
    torc_waitall();

    return true;
}

bool psrandom::setState()
{
    return init();
}

inline bool psrandom::setSeed(long const inSeed)
{
    std::lock_guard<std::mutex> lock(PRNG_m);
    if (!PRNG_initialized)
    {
        PRNG_seed = static_cast<std::size_t>(inSeed);
        return true;
    }
    // It has been initialized before
    return false;
}

inline double psrandom::unirnd()
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return uniformDoubleDistribution(NumberGenerator[me]);
}

inline double psrandom::unirnd(double const low, double const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return low + (high - low) * uniformDoubleDistribution(NumberGenerator[me]);
}

inline float psrandom::unirnd(float const low, float const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return low + (high - low) * uniformRealDistribution(NumberGenerator[me]);
}

inline void psrandom::unirnd(double *idata, int const nSize, double const low, double const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_real_distribution<double> d(low, high);
    for (auto i = 0; i < nSize; i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

inline void psrandom::unirnd(float *idata, int const nSize, float const low, float const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_real_distribution<float> d(low, high);
    for (auto i = 0; i < nSize; i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

inline void psrandom::unirnd(std::vector<double> &idata, double const low, double const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_real_distribution<double> d(low, high);
    for (auto i = 0; i < idata.size(); i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

inline void psrandom::unirnd(std::vector<float> &idata, float const low, float const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_real_distribution<float> d(low, high);
    for (auto i = 0; i < idata.size(); i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

inline void psrandom::u32rnd(int *idata, int const nSize, int const low, int const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_int_distribution<> d(low, high);
    for (auto i = 0; i < nSize; i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

inline void psrandom::u32rnd(std::vector<int> &idata, int const low, int const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_int_distribution<> d(low, high);
    for (auto i = 0; i < idata.size(); i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

inline double psrandom::drnd()
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].d();
}

inline double psrandom::drnd(double const low, double const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].d(low, high);
}

inline float psrandom::frnd()
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].f();
}

inline float psrandom::frnd(float const low, float const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].f(low, high);
}

inline unsigned int psrandom::u32()
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].u32();
}

inline unsigned int psrandom::u32(unsigned int const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].u32(high);
}

template <typename DataType>
inline void psrandom::shuffle(DataType *idata, int const nSize)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    for (int i = nSize - 1; i > 0; --i)
    {
        unsigned int const idx = saru[me].u32(i);
        std::swap(idata[i], idata[idx]);
    }
}

template <typename DataType>
inline void psrandom::shuffle(std::vector<DataType> &idata)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    for (auto i = idata.size() - 1; i > 0; --i)
    {
        unsigned int const idx = saru[me].u32(i);
        std::swap(idata[i], idata[idx]);
    }
}

} // namespace umuq

#endif // UMUQ_PSRANDOM
