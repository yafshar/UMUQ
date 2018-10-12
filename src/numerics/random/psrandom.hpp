#ifndef UMUQ_PSRANDOM_H
#define UMUQ_PSRANDOM_H

#include "../../core/core.hpp"
#include "../factorial.hpp"

/*! \defgroup Random_Module random distribution module
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
static std::vector<std::mt19937> NumberGenerator(1);

/*!
 * \ingroup Random_Module
 * 
 * \brief \c C++ Saru pseudo-random number generator.
 * 
 * \sa Saru.
 */
static std::vector<Saru> saru(1);

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
 * \tparam T Data type
 */
template <typename T>
static bool isPrngTaskRegistered = false;

/*!
 * \ingroup Random_Module
 * 
 * \brief Mutex object
 * 
 */
static std::mutex PRNG_m;

} // namespace umuq

#include "psrandom_normaldistribution.hpp"
#include "psrandom_lognormaldistribution.hpp"
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
 * \tparam T Data type one of float or double
 * 
 * \note
 * - Choosing the data type does not mean it only produces that type random number, the data type is only for function members.
 * 
 * - To use the psrandom in multithreaded application or in any class which requires setting the PRNG: <br>
 *   - First, construct a new psrandom object either with a seed or without it.
 *   - Second, initialize the PRNG or set the state of the PRNG.<br>
 *   - Third, use any member function or set the PRNG object in other classes 
 * 
 * \sa init.
 * \sa setState.
 * \sa umuq::density::densityFunction.
 */
template <typename T = double>
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

  private:
    // Make it noncopyable
    psrandom(psrandom const &) = delete;

    // Make it not assignable
    psrandom &operator=(psrandom const &) = delete;

  public:
    /*!
     * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
     *
     * \tparam D Input data type
     * 
     * \param idata Array of input data of type D
     * \param nSize Size of the array idata
     *
     * 
     * \note 
     * - This should be called after setting the State of psrandom object
     * 
     * 
     * The permutations generated by this algorithm occur with the same probability.
     *
     * Reference:<br>
     * R. Durstenfeld, "Algorithm 235: Random permutation" Communications of the ACM, 7 (1964), p. 420
     */
    template <typename D>
    inline void shuffle(D *idata, int const nSize);

    /*!
     * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
     *
     * \tparam D Input data type
     * 
     * \param idata Array of input data of type T
     * \param nSize Size of the array idata
     * 
     * \note 
     * - This can be called without setting the State of psrandom object
     */
    template <typename D>
    inline void Shuffle(D *idata, int const nSize);

  public:
    /*!
     * \returns Uniform random number between \f$ [a \cdots b) \f$
     * 
     * \brief Uniform random number between \f$ [a \cdots b) \f$
     *
     * Advance the PRNG state by 1, and output a T precision \f$ [a \cdots b) \f$ number (default \f$ a = 0, b = 1 \f$)
     */
    inline T unirnd(T const a = 0, T const b = 1);

    /*!
     * \returns a uniform random number of a double precision \f$ [0 \cdots 1) \f$ floating point 
     * 
     * \brief Advance state by 1, and output a double precision \f$ [0 \cdots 1) \f$ floating point
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline double drnd();

    /*!
     * \returns a uniform random number of a single precision \f$ [0 \cdots 1) \f$ floating point
     * 
     * \Advance state by 1, and output a single precision \f$ [0 \cdots 1) \f$ floating point
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline float frnd();

    /*!
     * \returns an unsigned 32 bit integer pseudo-random value
     * 
     * \brief Advance state by 1, and output a 32 bit integer pseudo-random value.
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline unsigned int u32rnd();

    /*! \fn multinomial
     * \brief The multinomial random distribution
     *  
     * \param p        Vector of probabilities \f$ p_1, \cdots, p_k \f$
     * \param K        Size of vector which shows K possible mutually exclusive outcomes 
     * \param N        N independent trials
     * \param mndist   A random sample from the multinomial distribution (with size of K)
     * 
     * 
     * \note 
     * - This should be called after setting the State of psrandom object
     * 
     * 
     * Let \f$ X=\left( X_1, \cdots, X_K \right) \f$ have a multinomial distribution \f$ M_K\left(N, p\right) \f$ <br>
     * The distribution of \f$ X \f$ is given by: <br>
     * \f$
     *     Pr(X_1=n_1, \cdots, X_K=n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \cdots p_K^{n_K}
     * \f$ <br> 
     *
     * where \f$ n_1, \cdots n_K \f$ are nonnegative integers satisfying \f$ sum_{i=1}^{K} {n_i} = N\f$, <br>
     * and \f$p = \left(p_1, \cdots, p_K\right)\f$ is a probability distribution. 
     *
     * Random variates are generated using the conditional binomial method. <br>
     * This scales well with N and does not require a setup step.
     *   
     *  Reference: <br>
     *  C.S. David, The computer generation of multinomial random variates, <br>
     *  Comp. Stat. Data Anal. 16 (1993) 205-217
     */
    bool multinomial(T const *p, int const K, int const N, int *mndist);

    /*! \fn Multinomial
     * \brief The multinomial distribution
     * 
     * \note 
     * - This can be called without setting the State of psrandom object
     */
    bool Multinomial(T const *p, int const K, int const N, int *mndist);

  public:
    /*!
     * \brief Replaces the normal object 
     * 
     * \param inMean    Input Mean
     * \param inStddev  Input standard deviation
     * 
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_normal(T const inMean, T const inStddev);
    inline bool set_normals(T const *inMean, T const *inStddev, int const N);
    inline bool set_normals(T const *inMeanInStddev, int const N);

    /*!
     * \brief Replaces the Normal object 
     * 
     * \param inMean    Input Mean
     * \param inStddev  Input standard deviation
     * 
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_Normal(T const inMean, T const inStddev);

    /*!
     * \brief Replaces the lnormal object 
     * 
     * \param inMean    Input Mean
     * \param inStddev  Input standard deviation
     * 
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_lnormal(T const inMean, T const inStddev);

    /*!
     * \brief Replaces the lNormal object 
     * 
     * \param inMean    Input Mean
     * \param inStddev  Input standard deviation
     * 
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_lNormal(T const inMean, T const inStddev);

    /*!
     * \brief Replaces the mvnormal object 
     * 
     * \param inMean        Mean vector of size \f$n\f$
     * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
     * 
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_mvnormal(EVectorX<T> const &inMean, EMatrixX<T> const &icovariance);

    /*!
     * \brief Replaces the mvnormal object
     *
     * \param inMean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \returns true 
     * \returns false in failure to allocate storage
     */

    inline bool set_mvnormal(T const *inMean, T const *icovariance, int const n);

    /*!
     * \brief Replaces the mvnormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     *
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_mvnormal(EMatrixX<T> const &icovariance);

    /*!
     * \brief Replaces the mvnormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \returns true 
     * \returns false in failure to allocate storage
     */

    inline bool set_mvnormal(T const *icovariance, int const n);

    /*!
     * \brief Replaces the mvnormal object (default mean = 0, covariance=I)
     *
     * \param n vector size
     *
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_mvnormal(int const n);

    /*!
     * \brief Replaces the mvNormal object 
     * 
     * \param inMean        Mean vector of size \f$n\f$
     * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
     * 
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_mvNormal(EVectorX<T> const &inMean, EMatrixX<T> const &icovariance);

    /*!
     * \brief Replaces the mvNormal object
     *
     * \param inMean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_mvNormal(T const *inMean, T const *icovariance, int const n);

    /*!
     * \brief Replaces the mvNormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     *
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_mvNormal(EMatrixX<T> const &icovariance);

    /*!
     * \brief Replaces the mvNormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_mvNormal(T const *icovariance, int const n);

    /*!
     * \brief Replaces the mvNormal object (default mean = 0, covariance=I)
     *
     * \param n vector size
     *
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_mvNormal(int const n);

    /*!
     * \brief Replaces the exponential object 
     * 
     * \param mu Mean, \f$ \mu \f$
     * 
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_expn(T const mu);
    inline bool set_expns(T const *mu, int const N);

    /*!
     * \brief Replaces the Gamma object 
     * 
     * \param alpha  Shape parameter \f$\alpha\f$
     * \param beta   Scale parameter \f$ beta\f$
     * 
     * \returns true 
     * \returns false in failure to allocate storage
     */
    inline bool set_gamma(T const alpha, T const beta);
    inline bool set_gammas(T const *alpha, T const *beta, int const N);
    inline bool set_gammas(T const *alphabeta, int const N);

  public:  
    /*!
     * \brief Normal (or Gaussian) random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::normalDistribution<T>> normal;
    std::unique_ptr<randomdist::normalDistribution<T>[]> normals;

    //! Number of normal distributions
    int nnormals;

    /*!
     * \brief Normal (or Gaussian) random number distribution
     * 
     * \note 
     * - This can be used without setting the State of psrandom object
     */
    std::unique_ptr<randomdist::NormalDistribution<T>> Normal;  
    
    /*!
     * \brief lognormal_distribution random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::lognormalDistribution<T>> lnormal;
   
    /*!
     * \brief lognormal_distribution random number distribution
     * 
     * \note 
     * - This can be used without setting the State of psrandom object
     */
    std::unique_ptr<randomdist::logNormalDistribution<T>> lNormal;

    /*!
     * \brief Multivariate random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::multivariatenormalDistribution<T>> mvnormal;
   
    /*!
     * \brief Multivariate random number distribution
     * 
     * \note 
     * -  This can be used without setting the State of psrandom object
     */
    std::unique_ptr<randomdist::multivariateNormalDistribution<T>> mvNormal;
    
    /*!
     * \brief Exponential random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::exponentialDistribution<T>> expn;
    std::unique_ptr<randomdist::exponentialDistribution<T>[]> expns;

    //! Number of Exponential distributions
    int nexpns;
   
    /*!
     * \brief Gamma random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::gammaDistribution<T>> gamma;
    std::unique_ptr<randomdist::gammaDistribution<T>[]> gammas;

    //! Number of Gamma distributions
    int ngammas;
};

template <typename T>
psrandom<T>::psrandom() : normal(nullptr),
                          normals(nullptr),
                          nnormals(0),
                          Normal(nullptr),
                          lnormal(nullptr),
                          lNormal(nullptr),
                          mvnormal(nullptr),
                          mvNormal(nullptr),
                          expn(nullptr),
                          expns(nullptr),
                          nexpns(0),
                          gamma(nullptr),
                          gammas(nullptr),
                          ngammas(0)
{
    if (!std::is_floating_point<T>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }
    {
        std::lock_guard<std::mutex> lock(PRNG_m);
        if (PRNG_seed == 0)
        {
            PRNG_seed = std::random_device{}();
        }

        if (!isPrngTaskRegistered<T>)
        {
            torc_register_task((void *)psrandom<T>::initTask);
            isPrngTaskRegistered<T> = true;
        }
    }
}

template <typename T>
psrandom<T>::psrandom(int const inSeed) : normal(nullptr),
                                          normals(nullptr),
                                          nnormals(0),
                                          Normal(nullptr),
                                          lnormal(nullptr),
                                          lNormal(nullptr),
                                          mvnormal(nullptr),
                                          mvNormal(nullptr),
                                          expn(nullptr),
                                          expns(nullptr),
                                          nexpns(0),
                                          gamma(nullptr),
                                          gammas(nullptr),
                                          ngammas(0)
{
    if (!std::is_floating_point<T>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }
    {
        std::lock_guard<std::mutex> lock(PRNG_m);
        if (PRNG_seed == 0)
        {
            PRNG_seed = static_cast<std::size_t>(inSeed);
        }

        if (!isPrngTaskRegistered<T>)
        {
            torc_register_task((void *)psrandom<T>::initTask);
            isPrngTaskRegistered<T> = true;
        }
    }
}

template <typename T>
psrandom<T>::~psrandom() {}

template <typename T>
void psrandom<T>::initTask()
{
    std::vector<std::size_t> rSeed(std::mt19937::state_size);

    // Get the local number of workers
    std::size_t nlocalworkers = static_cast<std::size_t>(torc_i_num_workers());

    // Node Id (MPI rank)
    std::size_t node_id = static_cast<std::size_t>(torc_node_id());

    std::size_t n = nlocalworkers * (node_id + 1);

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

template <typename T>
bool psrandom<T>::init()
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
        torc_create_ex(i * nlocalworkers, 1, (void (*)())psrandom<T>::initTask, 0);
    }
    torc_waitall();

    return true;
}

template <typename T>
bool psrandom<T>::setState()
{
    return init();
}

template <typename T>
inline bool psrandom<T>::setSeed(long const inSeed)
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

template <typename T>
inline T psrandom<T>::unirnd(T const a, T const b)
{
    UMUQFAIL("The Uniform random number of requested type is not implemented!");
}

template <>
inline double psrandom<double>::unirnd(double const a, double const b)
{
    // Get the thread ID
    int const me = torc_i_worker_id();
    return saru[me].d(a, b);
}

template <>
inline float psrandom<float>::unirnd(float a, float b)
{
    // Get the thread ID
    int const me = torc_i_worker_id();
    return saru[me].f(a, b);
}

template <typename T>
inline double psrandom<T>::drnd() { return saru[0].d(); }

template <typename T>
inline float psrandom<T>::frnd() { return saru[0].f(); }

template <typename T>
inline unsigned int psrandom<T>::u32rnd() { return saru[0].u32(); }

template <typename T>
template <typename D>
inline void psrandom<T>::shuffle(D *idata, int const nSize)
{
    // Get the thread ID
    int const me = torc_i_worker_id();

    for (int i = nSize - 1; i > 0; --i)
    {
        unsigned int const idx = saru[me].u32(i);
        std::swap(idata[i], idata[idx]);
    }
}

template <typename T>
template <typename D>
inline void psrandom<T>::Shuffle(D *idata, int const nSize)
{
    for (int i = nSize - 1; i > 0; --i)
    {
        unsigned int const idx = saru[0].u32(i);
        std::swap(idata[i], idata[idx]);
    }
}

template <typename T>
bool psrandom<T>::multinomial(T const *p, int const K, int const N, int *mndist)
{
    if (!PRNG_initialized)
    {
        UMUQFAILRETURN("One should set the current state of the engine before calling this function!");
    }

    // Get the thread ID
    int const me = torc_i_worker_id();

    T const totalProbabilitySum = std::accumulate(p, p + K, 0);

    T probabilitySum(0);
    int nProbabilitySum(0);
    for (int i = 0; i < K; i++)
    {
        if (p[i] > 0.0)
        {
            std::binomial_distribution<> d(N - nProbabilitySum, p[i] / (totalProbabilitySum - probabilitySum));
            mndist[i] = d(NumberGenerator[me]);
        }
        else
        {
            mndist[i] = 0;
        }
        probabilitySum += p[i];
        nProbabilitySum += mndist[i];
    }
    return true;
}

template <typename T>
bool psrandom<T>::Multinomial(T const *p, int const K, int const N, int *mndist)
{
    std::mt19937 gen(std::random_device{}());

    T const totalProbabilitySum = std::accumulate(p, p + K, 0);

    T probabilitySum(0);
    int nProbabilitySum(0);
    for (int i = 0; i < K; i++)
    {
        if (p[i] > 0.0)
        {
            std::binomial_distribution<> d(N - nProbabilitySum, p[i] / (totalProbabilitySum - probabilitySum));
            mndist[i] = d(gen);
        }
        else
        {
            mndist[i] = 0;
        }
        probabilitySum += p[i];
        nProbabilitySum += mndist[i];
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_normal(T const inMean, T const inStddev)
{
    try
    {
        normal.reset(new randomdist::normalDistribution<T>(inMean, inStddev));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_normals(T const *inMean, T const *inStddev, int const N)
{
    if (N > 0)
    {
        nnormals = N;
        try
        {
            normals.reset(new randomdist::normalDistribution<T>[nnormals]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0; i < nnormals; i++)
        {
            normals[i] = std::move(randomdist::normalDistribution<T>(inMean[i], inStddev[i]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename T>
inline bool psrandom<T>::set_normals(T const *inMeanInStddev, int const N)
{
    if (N > 0)
    {
        nnormals = N / 2;
        try
        {
            normals.reset(new randomdist::normalDistribution<T>[nnormals]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0, k = 0; i < nnormals; i++, k += 2)
        {
            normals[i] = std::move(randomdist::normalDistribution<T>(inMeanInStddev[k], inMeanInStddev[k + 1]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename T>
inline bool psrandom<T>::set_Normal(T const inMean, T const inStddev)
{
    try
    {
        Normal.reset(new randomdist::NormalDistribution<T>(inMean, inStddev));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_lnormal(T const inMean, T const inStddev)
{
    try
    {
        lnormal.reset(new randomdist::lognormalDistribution<T>(inMean, inStddev));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_lNormal(T const inMean, T const inStddev)
{
    try
    {
        lNormal.reset(new randomdist::logNormalDistribution<T>(inMean, inStddev));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvnormal(EVectorX<T> const &inMean, EMatrixX<T> const &icovariance)
{
    try
    {
        mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(inMean, icovariance));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvnormal(T const *inMean, T const *icovariance, int const n)
{
    try
    {
        mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(inMean, icovariance, n));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvnormal(EMatrixX<T> const &icovariance)
{
    try
    {
        mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(icovariance));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvnormal(T const *icovariance, int const n)
{
    try
    {
        mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(icovariance, n));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvnormal(int const n)
{
    try
    {
        mvnormal.reset(new randomdist::multivariatenormalDistribution<T>(n));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvNormal(EVectorX<T> const &inMean, EMatrixX<T> const &icovariance)
{
    try
    {
        mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(inMean, icovariance));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvNormal(T const *inMean, T const *icovariance, int const n)
{
    try
    {
        mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(inMean, icovariance, n));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvNormal(EMatrixX<T> const &icovariance)
{
    try
    {
        mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(icovariance));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvNormal(T const *icovariance, int const n)
{
    try
    {
        mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(icovariance, n));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_mvNormal(int const n)
{
    try
    {
        mvNormal.reset(new randomdist::multivariateNormalDistribution<T>(n));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_expn(T const mu)
{
    try
    {
        expn.reset(new randomdist::exponentialDistribution<T>(mu));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_expns(T const *mu, int const N)
{
    if (N > 0)
    {
        nexpns = N;
        try
        {
            expns.reset(new randomdist::exponentialDistribution<T>[nexpns]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0; i < nexpns; i++)
        {
            expns[i] = std::move(randomdist::exponentialDistribution<T>(mu[i]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename T>
inline bool psrandom<T>::set_gamma(T const alpha, T const beta)
{
    try
    {
        gamma.reset(new randomdist::gammaDistribution<T>(alpha, beta));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename T>
inline bool psrandom<T>::set_gammas(T const *alpha, T const *beta, int const N)
{
    if (N > 0)
    {
        ngammas = N;
        try
        {
            gammas.reset(new randomdist::gammaDistribution<T>[ngammas]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0; i < ngammas; i++)
        {
            gammas[i] = std::move(randomdist::gammaDistribution<T>(alpha[i], beta[i]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename T>
inline bool psrandom<T>::set_gammas(T const *alphabeta, int const N)
{
    if (N > 0)
    {
        ngammas = N / 2;
        try
        {
            gammas.reset(new randomdist::gammaDistribution<T>[ngammas]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0, k = 0; i < ngammas; i++, k += 2)
        {
            gammas[i] = std::move(randomdist::gammaDistribution<T>(alphabeta[k], alphabeta[k + 1]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

} // namespace umuq

#endif // UMUQ_PSRANDOM
