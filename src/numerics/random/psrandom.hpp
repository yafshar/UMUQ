#ifndef UMUQ_PSRANDOM_H
#define UMUQ_PSRANDOM_H

#include "../../core/core.hpp"
#include "../factorial.hpp"

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
 * \tparam RealType Data type
 */
template <typename RealType>
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
 * \brief Uniform random random of floating-point values uniformly distributed on the interval \f$ [0, 1) \f$ 
 * 
 * \tparam RealType Data type
 * 
 * \returns std::uniform_real_distribution<RealType> Uniform random random of floating-point values uniformly distributed on the interval \f$ [0, 1) \f$ 
 */
template <typename RealType>
std::uniform_real_distribution<RealType> uniformRealDistribution(RealType{}, RealType{1});

} // namespace umuq

#include "psrandom_uniformdistribution.hpp"
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
 * \tparam RealType Data type one of float or double
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
template <typename RealType = double>
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
     * Make it noncopyable.
     */
    psrandom(psrandom<RealType> const &) = delete;

    /*!
     * \brief Delete a psrandom object assignment
     * 
     * Make it nonassignable
     * 
     * \returns psrandom<RealType>& 
     */
    psrandom<RealType> &operator=(psrandom<RealType> const &) = delete;

  public:
    /*!
	 * \brief Uniform random of floating-point values uniformly distributed on 
	 * the interval \f$ [0, 1) \f$ using a random number engine based on Mersenne Twister
	 * 
	 * \return RealType A uniform random number between \f$ [0, 1) \f$
	 */
    inline RealType unirnd();

    /*!
	 * \brief Uniform random of floating-point values uniformly distributed on 
	 * the interval \f$ [a, b) \f$ using a random number engine based on Mersenne Twister
	 * 
	 * \param a  Lower bound of the interval (default is 0)
	 * \param b  Upper bound of theinterval  (default is 1)
	 * 
	 * \return RealType A uniform random number between \f$ [a, b) \f$
	 */
    inline RealType unirnd(RealType const a, RealType const b);

    /*!
	 * \brief Vector of uniform random of floating-point values uniformly distributed on 
	 * the interval \f$ [a, b) \f$ using a random number engine based on Mersenne Twister
	 * 
	 * \tparam RealType Data type 
	 * 
	 * \param idata  Array of input data of type D
	 * \param nSize  Size of the array 
	 * \param a      Lower bound of the interval (default is 0)
	 * \param b      Upper bound of theinterval  (default is 1)
	 */
    void unirnd(RealType *idata, int const nSize, RealType const a = RealType{}, RealType const b = RealType{1});

    /*!
	 * \brief Vector of uniform random of floating-point values uniformly distributed on 
	 * the interval \f$ [a, b) \f$ using a random number engine based on Mersenne Twister
	 * 
	 * \tparam RealType Data type 
	 * 
	 * \param idata  Array of input data of type D
	 * \param a      Lower bound of the interval (default is 0)
	 * \param b      Upper bound of theinterval  (default is 1)
	 */
    void unirnd(std::vector<RealType> &idata, RealType const a = RealType{}, RealType const b = RealType{1});

    /*!
	 * \brief Vector of uniform of integer values uniformly distributed on the closed interval
	 * \f$ [a, b] \f$ using a random number engine based on Mersenne Twister
	 * 
	 * \param idata  Array of input data of integers
	 * \param nSize  Size of the array 
	 * \param a      Lower bound of the interval 
	 * \param b      Upper bound of theinterval
	 */
    void u32rnd(int *idata, int const nSize, int const a, int const b);

    /*!
	 * \brief Vector of uniform of integer values uniformly distributed on the closed interval
	 * \f$ [a, b] \f$ using a random number engine based on Mersenne Twister
	 * 
	 * \param idata  Array of input data of integers
	 * \param a      Lower bound of the interval 
	 * \param b      Upper bound of theinterval
	 */
    void u32rnd(std::vector<int> &idata, int const a, int const b);

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
     * \brief Advance the Saru PRNG  state by 1, and output a single precision \f$ [0, 1) \f$ floating point
     * 
     * \returns A uniform random number of a single precision \f$ [0, 1) \f$ floating point
     * 
     * Reference:<br>
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, [Comput. Phys. Commun. 184, 1119-1128 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512003992).
     */
    inline float frnd();

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
     * \tparam D Input data type
     * 
     * \param idata Array of input data of type D
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
    template <typename D>
    inline void shuffle(D *idata, int const nSize);

    /*!
     * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
     *
     * \tparam D Input data type
     * 
     * \param idata Array of input data of type D
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
    template <typename D>
    inline void shuffle(std::vector<D> &idata);

  public:
    /*!
     * \brief Replaces the uniform object 
     * 
	 * \param a  Lower bound of the interval (default is 0)
	 * \param b  Upper bound of theinterval  (default is 1)
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_uniform(RealType const a = RealType{}, RealType const b = RealType{1});

    /*!
     * \brief Replaces the normal object 
     * 
	 * \param a  Lower bound of the interval (default is 0)
	 * \param b  Upper bound of theinterval  (default is 1)
     * \param N  Size of the arrays
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_uniforms(RealType const *a, RealType const *b, int const N);

    /*!
     * \brief Replaces the normal object 
     * 
	 * \param ab  Pair of lower and upper bound of the interval
     * \param N   Number of pairs of the lower and upper bound 
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_uniforms(RealType const *ab, int const N);

    /*!
     * \brief Replaces the normal object 
     * 
     * \param inMean    Input Mean
     * \param inStddev  Input standard deviation
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_normal(RealType const inMean, RealType const inStddev);

    /*!
     * \brief Replaces the normal object 
     * 
     * \param inMean    Input Mean
     * \param inStddev  Input standard deviation
     * \param N         Size of the means & standard deviation arrays
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_normals(RealType const *inMean, RealType const *inStddev, int const N);

    /*!
     * \brief Replaces the normal object 
     * 
     * \param inMeanInStddev  Pair of input Mean & standard deviation
     * \param N               Number of pairs of the means & standard deviation
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_normals(RealType const *inMeanInStddev, int const N);

    /*!
     * \brief Replaces the lnormal object 
     * 
     * \param inMean    Input Mean
     * \param inStddev  Input standard deviation
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_lnormal(RealType const inMean, RealType const inStddev);

    /*!
     * \brief Replaces the lnormal object 
     * 
     * \param inMean    Input Mean
     * \param inStddev  Input standard deviation
     * \param N         Size of the means & standard deviation arrays
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_lnormals(RealType const *inMean, RealType const *inStddev, int const N);

    /*!
     * \brief Replaces the lnormal object 
     * 
     * \param inMeanInStddev  Pair of input Mean & standard deviation
     * \param N               Number of pairs of the means & standard deviation
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_lnormals(RealType const *inMeanInStddev, int const N);

    /*!
     * \brief Replaces the mvnormal object 
     * 
     * \param inMean        Mean vector of size \f$n\f$
     * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_mvnormal(EVectorX<RealType> const &inMean, EMatrixX<RealType> const &icovariance);

    /*!
     * \brief Replaces the mvnormal object
     *
     * \param inMean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \returns false If it fails to allocate storage
     */
    inline bool set_mvnormal(RealType const *inMean, RealType const *icovariance, int const n);

    /*!
     * \brief Replaces the mvnormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     *
     * \returns false If it fails to allocate storage
     */
    inline bool set_mvnormal(EMatrixX<RealType> const &icovariance);

    /*!
     * \brief Replaces the mvnormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \returns false If it fails to allocate storage
     */
    inline bool set_mvnormal(RealType const *icovariance, int const n);

    /*!
     * \brief Replaces the mvnormal object (default mean = 0, covariance=I)
     *
     * \param n vector size
     *
     * \returns false If it fails to allocate storage
     */
    inline bool set_mvnormal(int const n);

    /*!
     * \brief Replaces the exponential object 
     * 
     * \param mu Mean, \f$ \mu \f$
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_expn(RealType const mu);

    /*!
     * \brief Replaces the exponential object 
     * 
     * \param mu Mean, \f$ \mu \f$
     * \param N  Number of means 
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_expns(RealType const *mu, int const N);

    /*!
     * \brief Replaces the Gamma object 
     * 
     * \param alpha  Shape parameter \f$ \alpha \f$
     * \param beta   Scale parameter \f$ \beta \f$
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_gamma(RealType const alpha, RealType const beta);
    /*!
     * \brief Replaces the Gamma object 
     * 
     * \param alpha  Shape parameter \f$\alpha \f$
     * \param beta   Scale parameter \f$ \beta \f$
     * \param N      Number of alphas and betas
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_gammas(RealType const *alpha, RealType const *beta, int const N);

    /*!
     * \brief Replaces the Gamma object 
     * 
     * \param alphabeta  Pair of Shape & Scale parameters \f$\alpha, \beta \f$
     * \param N          Number of pairs of Shape & Scale parameter
     * 
     * \returns false If it fails to allocate storage
     */
    inline bool set_gammas(RealType const *alphabeta, int const N);

  public:
    /*! 
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
    bool multinomial(RealType const *p, int const K, int const N, int *mndist);

  public:
    /*!
     * \brief Uniform random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::uniformDistribution<RealType>> uniform;

    /*!
     * \brief Uniform random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::uniformDistribution<RealType>[]> uniforms;

    /*! Number of uniform distributions. \sa uniforms */
    int nuniforms;

    /*!
     * \brief Normal (or Gaussian) random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::normalDistribution<RealType>> normal;

    /*!
     * \brief Normals (or Gaussian) random number distribution
     */
    std::unique_ptr<randomdist::normalDistribution<RealType>[]> normals;

    /*! Number of normal distributions. \sa normals */
    int nnormals;

    /*!
     * \brief lognormal_distribution random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::lognormalDistribution<RealType>> lnormal;

    /*!
     * \brief lognormal_distribution random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::lognormalDistribution<RealType>[]> lnormals;

    /*! Number of lnormal distributions. \sa lnormals */
    int nlnormals;

    /*!
     * \brief Multivariate random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::multivariateNormalDistribution<RealType>> mvnormal;

    /*!
     * \brief Exponential random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::exponentialDistribution<RealType>> expn;

    /*!
     * \brief Exponential random number distributions
     */
    std::unique_ptr<randomdist::exponentialDistribution<RealType>[]> expns;

    /*! Number of Exponential distributions \sa expns */
    int nexpns;

    /*!
     * \brief Gamma random number distribution
     * 
     * \note 
     * - This should be used after setting the State of psrandom object
     */
    std::unique_ptr<randomdist::gammaDistribution<RealType>> gamma;

    /*!
     * \brief Gamma random number distributions
     */
    std::unique_ptr<randomdist::gammaDistribution<RealType>[]> gammas;

    /*! Number of Gamma distributions \sa gammas */
    int ngammas;
};

template <typename RealType>
psrandom<RealType>::psrandom() : uniform(nullptr),
                                 uniforms(nullptr),
                                 nuniforms(0),
                                 normal(nullptr),
                                 normals(nullptr),
                                 nnormals(0),
                                 lnormal(nullptr),
                                 lnormals(nullptr),
                                 nlnormals(0),
                                 mvnormal(nullptr),
                                 expn(nullptr),
                                 expns(nullptr),
                                 nexpns(0),
                                 gamma(nullptr),
                                 gammas(nullptr),
                                 ngammas(0)
{
    if (!std::is_floating_point<RealType>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }
    {
        std::lock_guard<std::mutex> lock(PRNG_m);
        if (PRNG_seed == 0)
        {
            PRNG_seed = std::random_device{}();
        }

        if (!isPrngTaskRegistered<RealType>)
        {
            torc_register_task((void *)psrandom<RealType>::initTask);
            isPrngTaskRegistered<RealType> = true;
        }
    }
}

template <typename RealType>
psrandom<RealType>::psrandom(int const inSeed) : uniform(nullptr),
                                                 uniforms(nullptr),
                                                 nuniforms(0),
                                                 normal(nullptr),
                                                 normals(nullptr),
                                                 nnormals(0),
                                                 lnormal(nullptr),
                                                 lnormals(nullptr),
                                                 nlnormals(0),
                                                 mvnormal(nullptr),
                                                 expn(nullptr),
                                                 expns(nullptr),
                                                 nexpns(0),
                                                 gamma(nullptr),
                                                 gammas(nullptr),
                                                 ngammas(0)
{
    if (!std::is_floating_point<RealType>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }
    {
        std::lock_guard<std::mutex> lock(PRNG_m);
        if (PRNG_seed == 0)
        {
            PRNG_seed = static_cast<std::size_t>(inSeed);
        }

        if (!isPrngTaskRegistered<RealType>)
        {
            torc_register_task((void *)psrandom<RealType>::initTask);
            isPrngTaskRegistered<RealType> = true;
        }
    }
}

template <typename RealType>
psrandom<RealType>::~psrandom() {}

template <typename RealType>
void psrandom<RealType>::initTask()
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

template <typename RealType>
bool psrandom<RealType>::init()
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
        torc_create_ex(i * nlocalworkers, 1, (void (*)())psrandom<RealType>::initTask, 0);
    }
    torc_waitall();

    return true;
}

template <typename RealType>
bool psrandom<RealType>::setState()
{
    return init();
}

template <typename RealType>
inline bool psrandom<RealType>::setSeed(long const inSeed)
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

template <typename RealType>
inline RealType psrandom<RealType>::unirnd()
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return uniformRealDistribution<RealType>(NumberGenerator[me]);
}

template <typename RealType>
inline RealType psrandom<RealType>::unirnd(RealType const a, RealType const b)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return a + (b - a) * uniformRealDistribution<RealType>(NumberGenerator[me]);
}

template <typename RealType>
inline void psrandom<RealType>::unirnd(RealType *idata, int const nSize, RealType const a, RealType const b)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_real_distribution<RealType> d(a, b);
    for (auto i = 0; i < nSize; i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

template <typename RealType>
inline void psrandom<RealType>::unirnd(std::vector<RealType> &idata, RealType const a, RealType const b)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_real_distribution<RealType> d(a, b);
    for (auto i = 0; i < idata.size(); i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

template <typename RealType>
inline void psrandom<RealType>::u32rnd(int *idata, int const nSize, int const a, int const b)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_int_distribution<> d(a, b);
    for (auto i = 0; i < nSize; i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

template <typename RealType>
inline void psrandom<RealType>::u32rnd(std::vector<int> &idata, int const a, int const b)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    std::uniform_int_distribution<> d(a, b);
    for (auto i = 0; i < idata.size(); i++)
    {
        idata[i] = d(NumberGenerator[me]);
    }
}

template <typename RealType>
inline double psrandom<RealType>::drnd()
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].d();
}

template <typename RealType>
inline float psrandom<RealType>::frnd()
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].f();
}

template <typename RealType>
inline unsigned int psrandom<RealType>::u32()
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].u32();
}

template <typename RealType>
inline unsigned int psrandom<RealType>::u32(unsigned int const high)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    return saru[me].u32(high);
}

template <typename RealType>
template <typename D>
inline void psrandom<RealType>::shuffle(D *idata, int const nSize)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    for (int i = nSize - 1; i > 0; --i)
    {
        unsigned int const idx = saru[me].u32(i);
        std::swap(idata[i], idata[idx]);
    }
}

template <typename RealType>
template <typename D>
inline void psrandom<RealType>::shuffle(std::vector<D> &idata)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;
    for (auto i = idata.size() - 1; i > 0; --i)
    {
        unsigned int const idx = saru[me].u32(i);
        std::swap(idata[i], idata[idx]);
    }
}

template <typename RealType>
inline bool psrandom<RealType>::set_uniform(RealType const a, RealType const b)
{
    try
    {
        uniform.reset(new randomdist::uniformDistribution<RealType>(a, b));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_uniforms(RealType const *a, RealType const *b, int const N)
{
    if (N > 0)
    {
        nuniforms = N;
        try
        {
            uniforms.reset(new randomdist::uniformDistribution<RealType>[nuniforms]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0; i < nuniforms; i++)
        {
            uniforms[i] = std::move(randomdist::uniformDistribution<RealType>(a[i], b[i]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename RealType>
inline bool psrandom<RealType>::set_uniforms(RealType const *ab, int const N)
{
    if (N > 0)
    {
        nuniforms = N / 2;
        try
        {
            uniforms.reset(new randomdist::uniformDistribution<RealType>[nuniforms]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0, k = 0; i < nuniforms; i++, k += 2)
        {
            uniforms[i] = std::move(randomdist::uniformDistribution<RealType>(ab[k], ab[k + 1]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename RealType>
inline bool psrandom<RealType>::set_normal(RealType const inMean, RealType const inStddev)
{
    try
    {
        normal.reset(new randomdist::normalDistribution<RealType>(inMean, inStddev));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_normals(RealType const *inMean, RealType const *inStddev, int const N)
{
    if (N > 0)
    {
        nnormals = N;
        try
        {
            normals.reset(new randomdist::normalDistribution<RealType>[nnormals]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0; i < nnormals; i++)
        {
            normals[i] = std::move(randomdist::normalDistribution<RealType>(inMean[i], inStddev[i]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename RealType>
inline bool psrandom<RealType>::set_normals(RealType const *inMeanInStddev, int const N)
{
    if (N > 0)
    {
        nnormals = N / 2;
        try
        {
            normals.reset(new randomdist::normalDistribution<RealType>[nnormals]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0, k = 0; i < nnormals; i++, k += 2)
        {
            normals[i] = std::move(randomdist::normalDistribution<RealType>(inMeanInStddev[k], inMeanInStddev[k + 1]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename RealType>
inline bool psrandom<RealType>::set_lnormal(RealType const inMean, RealType const inStddev)
{
    try
    {
        lnormal.reset(new randomdist::lognormalDistribution<RealType>(inMean, inStddev));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_lnormals(RealType const *inMean, RealType const *inStddev, int const N)
{
    if (N > 0)
    {
        nlnormals = N;
        try
        {
            lnormals.reset(new randomdist::lognormalDistribution<RealType>[nlnormals]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0; i < nlnormals; i++)
        {
            lnormals[i] = std::move(randomdist::lognormalDistribution<RealType>(inMean[i], inStddev[i]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename RealType>
inline bool psrandom<RealType>::set_lnormals(RealType const *inMeanInStddev, int const N)
{
    if (N > 0)
    {
        nlnormals = N / 2;
        try
        {
            lnormals.reset(new randomdist::lognormalDistribution<RealType>[nnormals]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0, k = 0; i < nlnormals; i++, k += 2)
        {
            lnormals[i] = std::move(randomdist::lognormalDistribution<RealType>(inMeanInStddev[k], inMeanInStddev[k + 1]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename RealType>
inline bool psrandom<RealType>::set_mvnormal(EVectorX<RealType> const &inMean, EMatrixX<RealType> const &icovariance)
{
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(inMean, icovariance));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_mvnormal(RealType const *inMean, RealType const *icovariance, int const n)
{
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(inMean, icovariance, n));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_mvnormal(EMatrixX<RealType> const &icovariance)
{
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(icovariance));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_mvnormal(RealType const *icovariance, int const n)
{
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(icovariance, n));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_mvnormal(int const n)
{
    try
    {
        mvnormal.reset(new randomdist::multivariateNormalDistribution<RealType>(n));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_expn(RealType const mu)
{
    try
    {
        expn.reset(new randomdist::exponentialDistribution<RealType>(mu));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_expns(RealType const *mu, int const N)
{
    if (N > 0)
    {
        nexpns = N;
        try
        {
            expns.reset(new randomdist::exponentialDistribution<RealType>[nexpns]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0; i < nexpns; i++)
        {
            expns[i] = std::move(randomdist::exponentialDistribution<RealType>(mu[i]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename RealType>
inline bool psrandom<RealType>::set_gamma(RealType const alpha, RealType const beta)
{
    try
    {
        gamma.reset(new randomdist::gammaDistribution<RealType>(alpha, beta));
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }
    return true;
}

template <typename RealType>
inline bool psrandom<RealType>::set_gammas(RealType const *alpha, RealType const *beta, int const N)
{
    if (N > 0)
    {
        ngammas = N;
        try
        {
            gammas.reset(new randomdist::gammaDistribution<RealType>[ngammas]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0; i < ngammas; i++)
        {
            gammas[i] = std::move(randomdist::gammaDistribution<RealType>(alpha[i], beta[i]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename RealType>
inline bool psrandom<RealType>::set_gammas(RealType const *alphabeta, int const N)
{
    if (N > 0)
    {
        ngammas = N / 2;
        try
        {
            gammas.reset(new randomdist::gammaDistribution<RealType>[ngammas]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        for (int i = 0, k = 0; i < ngammas; i++, k += 2)
        {
            gammas[i] = std::move(randomdist::gammaDistribution<RealType>(alphabeta[k], alphabeta[k + 1]));
        }
        return true;
    }
    UMUQFAILRETURN("Wrong number of distributions requested!");
}

template <typename RealType>
bool psrandom<RealType>::multinomial(RealType const *p, int const K, int const N, int *mndist)
{
    // Get the thread ID
    int const me = PRNG_initialized ? torc_i_worker_id() : 0;

    RealType const totalProbabilitySum = std::accumulate(p, p + K, 0);

    RealType probabilitySum(0);
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

} // namespace umuq

#endif // UMUQ_PSRANDOM
