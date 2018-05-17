#ifndef UMUQ_PSRANDOM_H
#define UMUQ_PSRANDOM_H

#include "saruprng.hpp"
#include "factorial.hpp"

/*! \class psrandom
  *
  * This class generates pseudo-random numbers.
  * Engines and distributions used to produce random values. 
  * 
  * All of the engines may be specifically seeded, for use with repeatable simulators.
  * Random number engines generate pseudo-random numbers using seed data as entropy source. 
  * The choice of which engine to use involves a number of tradeoffs: 
  * 
  * Saru PRNG has only a small storage requirement for state which is 64-bit and is very fast. 
  * 
  * The Mersenne twister is slower and has greater state storage requirements but with the right parameters has 
  * the longest non-repeating sequence with the most desirable spectral characteristics (for a given definition of desirable). 
  */
struct psrandom
{
    /*!
     * \brief Default constructor
     */
    psrandom()
    {
        if (iseed != 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "There should only be one instance of a psrandom object!" << std::endl;
            throw(std::runtime_error("There should only be one instance of a psrandom object!"));
        }

        iseed = std::random_device{}();
    };

    /*!
     * \brief constructor
     * 
     * \param iseed_ Input seed for random number initialization 
     */
    psrandom(size_t const &iseed_)
    {
        if (iseed != 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "There should only be one instance of a psrandom object!" << std::endl;
            throw(std::runtime_error("There should only be one instance of a psrandom object!"));
        }

        iseed = iseed_;

        try
        {
            //Number of local workers
            int const nlocalworkers = torc_i_num_workers();

            NumberGenerator = new std::mt19937[nlocalworkers];
            saru = new Saru[nlocalworkers];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        };
    }

    /*!
     *  \brief destructor 
     *    
     */
    ~psrandom()
    {
        destroy();
    };

    /*!
     * \brief destroy the created memory and set the static variable to NULL
     *
     */
    void destroy()
    {
        iseed = 0;

        if (NumberGenerator != nullptr)
        {
            delete[] NumberGenerator;
            NumberGenerator = nullptr;
        }

        if (saru != nullptr)
        {
            delete[] saru;
            saru = nullptr;
        }
    }

    /*!
     * \brief Init task on each node to set the current state of the engine for all the threads on that node 
     */
    static void init_Task();

    /*!
     * \returns \a true when sets the current state of the engine successfully
     * 
     * \brief Sets the current state of the engine
     */
    bool init();

    /*!
     * \returns Uniform random number between \f$ [a \cdots b) \f$
     * 
     * \brief Uniform random number between \f$ [a \cdots b) \f$
     *
     * \tparam T data type one of float or double
     *
     * Advance the PRNG state by 1, and output a T precision \f$ [a \cdots b) \f$ number (default \f$ a = 0, b = 1 \f$)
     */
    template <typename T>
    inline T unirnd(T const a = 0, T const b = 1)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " The Uniform random number of type " << typeid(T).name() << " is not implemented !" << std::endl;
        throw(std::runtime_error("Wrong type!"));
    }

    /*!
     * \returns a uniform random number of a double precision \f$ [0 \cdots 1) \f$ floating point 
     * 
     * \brief Advance state by 1, and output a double precision \f$ [0 \cdots 1) \f$ floating point
     * 
     * Reference:
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
     */
    inline double d() { return saru[0].d(); }

    /*!
     * \returns a uniform random number of a single precision \f$ [0 \cdots 1) \f$ floating point
     * 
     * \Advance state by 1, and output a single precision \f$ [0 \cdots 1) \f$ floating point
     * 
     * Reference:
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
     */
    inline float f() { return saru[0].f(); }

    /*!
     * \returns an unisgned 32 bit integer pseudo-random value
     * 
     * \brief Advance state by 1, and output a 32 bit integer pseudo-random value.
     * 
     * Reference:
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
     */
    inline unsigned int u32() { return saru[0].u32(); }

    /*!
     * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
     *
     * \param idata array of input data of type int
     * \param nSize Size of the array idata
     *
     * The permutations generated by this algorithm occur with the same probability.
     *
     * References : 
     * R. Durstenfeld, "Algorithm 235: Random permutation" Communications of the ACM, 7 (1964), p. 420
     */
    template <typename T = int>
    void shuffle(T *idata, int const nSize)
    {
        //Get the thread ID
        int const me = torc_i_worker_id();

        for (int i = nSize - 1; i > 0; --i)
        {
            unsigned int const idx = saru[me].u32(i);
            std::swap(idata[i], idata[idx]);
        }
    }

    template <typename T = int>
    void Shuffle(T *idata, int const nSize)
    {
        for (int i = nSize - 1; i > 0; --i)
        {
            unsigned int const idx = saru[0].u32(i);
            std::swap(idata[i], idata[idx]);
        }
    }

    //! RNG seed
    static size_t iseed;

    //! 32-bit Mersenne Twister by Matsumoto and Nishimura, 1998
    static std::mt19937 *NumberGenerator;

    //! C++ Saru PRNG
    static Saru *saru;
};

size_t psrandom::iseed = 0;
std::mt19937 *psrandom::NumberGenerator = nullptr;
Saru *psrandom::saru = nullptr;

/*!
 * \brief init Task on each node to set the current state of the engine
 */
void psrandom::init_Task()
{
    size_t rseed[std::mt19937::state_size];

    //Get the local number of workers
    size_t nlocalworkers = (size_t)torc_i_num_workers();

    //Node Id (MPI rank)
    size_t node_id = (size_t)torc_node_id();

    size_t n = nlocalworkers * (node_id + 1);

    for (size_t i = 0; i < nlocalworkers; i++)
    {
        size_t const j = psrandom::iseed + n + i;

        for (size_t k = 0; k < std::mt19937::state_size; k++)
        {
            rseed[k] = k + j;
        }

        //Seed the engine with unsigned ints
        std::seed_seq sseq(rseed, rseed + std::mt19937::state_size);

        //For each thread feed the RNG
        psrandom::NumberGenerator[i].seed(sseq);

        Saru s(psrandom::iseed, n, i);
        psrandom::saru[i] = std::move(s);
    }
}

/*!
 * \brief Set the current state of the engine
 */
bool psrandom::init()
{
    //Make sure MPI is initilized
    auto initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to initilize MPI " << std::endl;
        return false;
    }

    torc_register_task((void *)psrandom::init_Task);

    int nlocalworkers = torc_i_num_workers();

    if (psrandom::NumberGenerator == nullptr)
    {
        try
        {
            psrandom::NumberGenerator = new std::mt19937[nlocalworkers];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        };
    }
    if (psrandom::saru == nullptr)
    {
        try
        {
            psrandom::saru = new Saru[nlocalworkers];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        };
    }

    for (int i = 0; i < torc_num_nodes(); i++)
    {
        torc_create_ex(i * nlocalworkers, 1, (void (*)())psrandom::init_Task, 0);
    }
    torc_waitall();

    return true;
}

/*!
 * \brief Uniform random number between \f$ [a \cdots b) \f$
 *
 * \tparam T data type one of float, double
 *
 * Advance the PRNG state by 1, and output a T precision \f$ [a \cdots b) \f$ number (default \f$ a = 0 , b = 1 \f$)
 * This is a partial specialization to make a special case for double precision uniform random number
 */
template <>
inline double psrandom::unirnd<double>(double const a, double const b)
{
    //Get the thread ID
    int const me = torc_i_worker_id();
    return saru[me].d(a, b);
}

/*!
 * \brief This is a partial specialization to make a special case for float precision uniform random number between \f$ [a \cdots b) \f$
 * \param a lower bound 
 * \param b upper bound
 */
template <>
inline float psrandom::unirnd<float>(float a, float b)
{
    //Get the thread ID
    int const me = torc_i_worker_id();
    return saru[me].f(a, b);
}

/*! \fn multinomial
 * \brief The multinomial distribution
 * 
 * This is based on psrandom object seeded engine.  
 * So to use it there should be an instance of psrandom object.
 * 
 * \tparam   T       data type 
 * \param    p       vector of probabilities \f$ p_1, \cdots, p_k \f$
 * \param    K       size of vector which shows K possible mutually exclusive outcomes 
 * \param    N       N independent trials
 * \param    mndist  A random sample from the multinomial distribution
 * 
 * 
 * Let \f$ X=\left( X_1, \cdots, X_K \right) \f$ have a multinomial distribution \f$ M_K\left(N, p\right) \f$
 * The distribution of \f$ X \f$ is given by:
 * \f[
 *     Pr(X_1=n_1, \cdots, X_K=n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \cdots p_K^{n_K}
 * \f] 
 *
 * where \f$ n_1, \cdots n_K \f$ are nonnegative integers satisfying \f$ sum_{i=1}^{K} {n_i} = N\f$,
 * and \f$p = \left(p_1, \cdots, p_K\right)\f$ is a probability distribution. 
 *
 * Random variates are generated using the conditional binomial method.
 * This scales well with N and does not require a setup step.
 *   
 *  Ref: 
 *  C.S. David, The computer generation of multinomial random variates,
 *  Comp. Stat. Data Anal. 16 (1993) 205-217
 */
template <typename T = double>
bool multinomial(T const *p, unsigned int const K, unsigned int const N, unsigned int *mndist)
{
    if (psrandom::iseed == 0)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "There should be an instance of a psrandom object before using this class!" << std::endl;
        return false;
    }

    //Get the thread ID
    int const me = torc_i_worker_id();

    T const totpsum = std::accumulate(p, p + K, 0);

    T psum = 0;
    unsigned int nsum = 0;
    for (int i = 0; i < K; i++)
    {
        if (p[i] > 0.0)
        {
            std::binomial_distribution<> d(N - nsum, p[i] / (totpsum - psum));
            mndist[i] = d(psrandom::NumberGenerator[me]);
        }
        else
        {
            mndist[i] = 0;
        }
        psum += p[i];
        nsum += mndist[i];
    }
    return true;
}

/*! \fn Multinomial
 * \brief The Multinomial distribution
 * 
 * This is independent. 
 * 
 * \tparam   T       data type 
 * \param    p       vector of probabilities \f$ p_1, \cdots, p_k \f$
 * \param    K       size of vector which shows K possible mutually exclusive outcomes 
 * \param    N       N independent trials
 * \param    mndist  A random sample from the multinomial distribution
 * 
 * 
 * Let \f$ X=\left( X_1, \cdots, X_K \right) \f$ have a multinomial distribution \f$ M_K\left(N, p\right) \f$
 * The distribution of \f$ X \f$ is given by:
 * \f[
 *     Pr(X_1=n_1, \cdots, X_K=n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \cdots p_K^{n_K}
 * \f] 
 *
 * where \f$ n_1, \cdots n_K \f$ are nonnegative integers satisfying \f$ sum_{i=1}^{K} {n_i} = N\f$,
 * and \f$p = \left(p_1, \cdots, p_K\right)\f$ is a probability distribution. 
 *
 * Random variates are generated using the conditional binomial method.
 * This scales well with N and does not require a setup step.
 *   
 *  Ref: 
 *  C.S. David, The computer generation of multinomial random variates,
 *  Comp. Stat. Data Anal. 16 (1993) 205-217
 */
template <typename T = double>
bool Multinomial(T const *p, unsigned int const K, unsigned int const N, unsigned int *mndist)
{
    std::mt19937 gen(std::random_device{}());

    T const totpsum = std::accumulate(p, p + K, 0);

    T psum = 0;
    unsigned int nsum = 0;
    for (int i = 0; i < K; i++)
    {
        if (p[i] > 0.0)
        {
            std::binomial_distribution<> d(N - nsum, p[i] / (totpsum - psum));
            mndist[i] = d(gen);
        }
        else
        {
            mndist[i] = 0;
        }
        psum += p[i];
        nsum += mndist[i];
    }
    return true;
}

/*! \fn multinomial_lnpdf
 * \brief Computes the logarithm of the probability from the multinomial distribution
 * 
 * This function computes the logarithm of the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
 * from a multinomial distribution with probabilities \f$p[K]\f$.
 * 
 * \tparam T      data type one of float or double
 * \param  mndist A random sample (with size of K) from the multinomial distribution
 * \param  p      vector of probabilities \f$ p_1, \cdots, p_k \f$
 * \param  K      size of vector
 * 
 * \returns the logarithm of the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
 */
template <typename T = double>
T multinomial_lnpdf(unsigned int const *mndist, T const *p, int const K)
{
    // compute the total number of independent trials
    unsigned int N = std::accumulate(mndist, mndist + K, 0);

    T const totpsum = std::accumulate(p, p + K, 0);

    //Currently we have the limitation of float or double type in factorial implementation
    T log_pdf = factorial<T>(N);

    for (int i = 0; i < K; i++)
    {
        if (mndist[i] > 0)
        {
            log_pdf += std::log(p[i] / totpsum) * mndist[i] - factorial<T>(mndist[i]);
        }
    }

    return log_pdf;
}

/*! \fn multinomial_pdf
 * \brief Computes the probability from the multinomial distribution
 * 
 * This function computes the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
 * from a multinomial distribution with probabilities \f$p[K]\f$.
 * 
 * \tparam T      data type one of float or double
 * \param  mndist A random sample (with size of K) from the multinomial distribution
 * \param  p      vector of probabilities \f$ p_1, \cdots, p_k \f$
 * \param  K      size of vector
 * 
 * \returns the probability \f$Pr(X_1=n_1, \cdots, X_K=n_K)\f$ of sampling \f$n[K]\f$ 
 */
template <typename T = double>
T multinomial_pdf(unsigned int const *mndist, T const *p, int const K)
{
    // compute the total number of independent trials
    unsigned int N = std::accumulate(mndist, mndist + K, 0);

    T const totpsum = std::accumulate(p, p + K, 0);

    T log_pdf = factorial<T>(N);

    for (int i = 0; i < K; i++)
    {
        if (mndist[i] > 0)
        {
            log_pdf += std::log(p[i] / totpsum) * mndist[i] - factorial<T>(mndist[i]);
        }
    }

    return std::exp(log_pdf);
}

/*! \class normrnd
 * \brief Generates random numbers according to the Normal (or Gaussian) random number distribution
 * 
 * This class is based on psrandom object seeded engine. So to use this object there should be an instance of 
 * psrandom object.
 *
 * \tparam T data type one of float, double, or long double
 *
 */
template <typename T = double>
class normrnd
{
  public:
    /*!
     * \brief Default constructor (default mean = 0, stddev = 1)
     */
    normrnd(T mean = 0, T stddev = 1) : d(mean, stddev)
    {
        if (psrandom::iseed == 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "There should be an instance of a psrandom object before using this class!" << std::endl;
            throw(std::runtime_error("There should be an instance of a psrandom object before using this class!"));
        }
    }

    /*!
     * \returns random numbers x according to Normal (or Gaussian) random number distribution
     * The result type generated by the generator is undefined if @T is not one of float, double, or long double
     */
    T operator()()
    {
        //Get the thread ID
        int const me = torc_i_worker_id();
        return d(psrandom::NumberGenerator[me]);
    }

  private:
    //! Random numbers according to the Normal (or Gaussian) random number distribution
    std::normal_distribution<T> d;
};

/*! \class Normrnd
 * \brief Generates random numbers according to the Normal (or Gaussian) random number distribution
 * This class is independent.
 *
 * \tparam T data type one of float, double, or long double
 *
 */
template <typename T = double>
class Normrnd
{
  public:
    /*!
     * \brief Default constructor (default mean = 0, stddev = 1)
     */
    Normrnd(T mean = 0, T stddev = 1) : d(mean, stddev), gen(std::random_device{}()) {}

    /*!
     * \returns random numbers x according to Normal (or Gaussian) random number distribution
     * The result type generated by the generator is undefined if @T is not one of float, double, or long double
     */
    T operator()() { return d(gen); }

  private:
    //! Random number engine based on Mersenne Twister algorithm.
    std::mt19937 gen;

    //! Random numbers according to the Normal (or Gaussian) random number distribution
    std::normal_distribution<T> d;
};

/*! \class lognormrnd
 * \brief Generates random numbers x > 0 according to the lognormal_distribution
 * This class is based on psrandom object seeded engine. So to use this object there should be an instance of 
 * psrandom object.
 * 
 * \tparam T data type one of float, double, or long double
 *
 */
template <typename T = double>
class lognormrnd
{
  public:
    /*!
     * \brief Default constructor (default mean = 0, stddev = 1)
     */
    lognormrnd(T mean = 0, T stddev = 1) : d(mean, stddev)
    {
        if (psrandom::iseed == 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "There should be an instance of a psrandom object before using this class!" << std::endl;
            throw(std::runtime_error("There should be an instance of a psrandom object before using this class!"));
        }
    }

    /*!
     * \returns random numbers x > 0 according to the lognormal_distribution
     * The result type generated by the generator is undefined if @T is not one of float, double, or long double
     */
    T operator()()
    {
        //Get the thread ID
        int const me = torc_i_worker_id();
        return d(psrandom::NumberGenerator[me]);
    }

  private:
    //! Lognormal_distribution random number distribution
    std::lognormal_distribution<T> d;
};

/*! \class Lognormrnd
 * \brief Generates random numbers \f$ x > 0 \f$ according to the lognormal_distribution
 * This class is independent.
 * 
 * \tparam T data type one of float, double, or long double
 *
 */
template <typename T = double>
class Lognormrnd
{
  public:
    /*!
     * \brief Default constructor (default mean = 0, stddev = 1)
     */
    Lognormrnd(T mean = 0, T stddev = 1) : d(mean, stddev), gen(std::random_device{}()) {}

    /*!
     * \returns random numbers x > 0 according to the lognormal_distribution
     * The result type generated by the generator is undefined if @T is not one of float, double, or long double
     */
    T operator()() { return d(gen); }

  private:
    //! Random number engine based on Mersenne Twister algorithm.
    std::mt19937 gen;

    //! Lognormal_distribution random number distribution
    std::lognormal_distribution<T> d;
};

/*! \class mvnormdist
 * \brief Multivariate normal distribution
 * This class is based on psrandom object seeded engine. 
 * So to use this object there should be an instance of psrandom object.
 *
 * 
 * \tparam T data type one of float, double, or long double
 *
 */
template <typename T>
class mvnormdist
{
  public:
    /*!
     * \brief constructor
     *
     * \param mean_       mean vector of size \f$n\f$
     * \param covariance_ variance-covariance matrix of size \f$n \times n\f$
     */
    mvnormdist(EVectorX<T> const &mean_, EMatrixX<T> const &covariance_) : mean(mean_),
                                                                           covariance(covariance_),
                                                                           lu(covariance_)
    {
        if (psrandom::iseed == 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "There should be an instance of a psrandom object before using this class!" << std::endl;
            throw(std::runtime_error("There should be an instance of a psrandom object before using this class!"));
        }

        // Computes eigenvalues and eigenvectors of selfadjoint matrices.
        Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
        transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    }

    /*!
     * \brief constructor
     * 
     * \param mean_       mean vector of size \f$n\f$
     * \param covariance_ variance-covariance matrix of size \f$n \times n\f$
     * \param n           vector size
     */
    mvnormdist(T const *mean_, T const *covariance_, int const n) : mean(CTEMapX<T>(mean_, n, 1)),
                                                                    covariance(CTEMapX<T>(covariance_, n, n)),
                                                                    lu(CTEMapX<T>(covariance_, n, n))
    {
        if (psrandom::iseed == 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "There should be an instance of a psrandom object before using this class!" << std::endl;
            throw(std::runtime_error("There should be an instance of a psrandom object before using this class!"));
        }

        // Computes eigenvalues and eigenvectors of selfadjoint matrices.
        Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
        transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    }

    /*!
     * \brief constructor (default mean = 0)
     *
     * \param covariance_ variance-covariance matrix of size \f$n \times n\f$
     */
    mvnormdist(EMatrixX<T> const &covariance_) : mvnormdist(EVectorX<T>::Zero(covariance_.rows()), covariance_) {}

    /*!
     * \brief constructor (default mean = 0)
     * 
     * \param covariance_ variance-covariance matrix of size \f$n \times n\f$
     * \param n           vector size
     */
    mvnormdist(T const *covariance_, int const n) : mean(EVectorX<T>::Zero(n)),
                                                    covariance(CTEMapX<T>(covariance_, n, n)),
                                                    lu(CTEMapX<T>(covariance_, n, n))
    {
        if (psrandom::iseed == 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "There should be an instance of a psrandom object before using this class!" << std::endl;
            throw(std::runtime_error("There should be an instance of a psrandom object before using this class!"));
        }

        // Computes eigenvalues and eigenvectors of selfadjoint matrices.
        Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
        transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    }

    /*!
     * \brief constructor (default mean = 0, covariance=I)
     * 
     * \param n vector size
     */
    mvnormdist(int const n) : mean(EVectorX<T>::Zero(n)),
                              covariance(EMatrixX<T>::Identity(n, n)),
                              transform(EMatrixX<T>::Identity(n, n)),
                              lu(EMatrixX<T>::Identity(n, n))
    {
        if (psrandom::iseed == 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "There should be an instance of a psrandom object before using this class!" << std::endl;
            throw(std::runtime_error("There should be an instance of a psrandom object before using this class!"));
        }
    }

    /*!
     * \returns a vector with multivariate normal distribution
     */
    EVectorX<T> operator()()
    {
        int const me = torc_i_worker_id();
        return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T x) { return d(psrandom::NumberGenerator[me]); });
    }

    /*!
     * \brief PDF
     * 
     * \param X vector of size \f$ n \f$
     * 
     * \returns pdf of X
     */
    T pdf(EVectorX<T> const &X)
    {
        T denom = std::pow(M_2PI, X.rows()) * lu.determinant();

        EVectorX<T> ax = X - mean;

        //Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T MDistSq = ax.transpose() * lu.inverse() * ax;

        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }

    /*!
     * \brief PDF
     * 
     * \param x vector of size \f$ n \f$
     * \param n vector size
     * 
     * \returns pdf of x
     */
    T pdf(T const *x, int const n)
    {
        CTEMapX<T, Eigen::ColMajor> X(x, n, 1);

        T denom = std::pow(M_2PI, n) * lu.determinant();

        EVectorX<T> ax = X - mean;

        //Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T MDistSq = ax.transpose() * lu.inverse() * ax;

        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }

    /*!
     * \brief LOGPDF
     * 
     * \param X vector of size \f$ n \f$
     * 
     * \returns logpdf of X
     */
    T lnpdf(EVectorX<T> const &X)
    {
        EVectorX<T> ax = X - mean;

        //Mahalanobis distance between \f$ X \f$ and \f$\ mu \f$
        T MDistSq = ax.transpose() * lu.inverse() * ax;

        return -0.5 * (MDistSq + X.rows() * M_L2PI + std::log(lu.determinant()));
    }

    /*!
     * \brief LOGPDF
     * 
     * \param x vector of size \f$ n \f$
     * \param n vector size
     * 
     * \returns logpdf of x
     */
    T lnpdf(T const *x, int const n)
    {
        CTEMapX<T, Eigen::ColMajor> X(x, n, 1);

        EVectorX<T> ax = X - mean;

        //Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T MDistSq = ax.transpose() * lu.inverse() * ax;

        return -0.5 * (MDistSq + n * M_L2PI + std::log(lu.determinant()));
    }

  private:
    //! Vector of size \f$n\f$
    EVectorX<T> mean;
    
    //! Variance-covariance matrix of size \f$ n \times n \f$
    EMatrixX<T> covariance;
    
    //! Matrix of size \f$n \times n\f$
    EMatrixX<T> transform;
    
    //! LU decomposition of a matrix with complete pivoting
    Eigen::FullPivLU<EMatrixX<T>> lu;
    
    //! Generates random numbers according to the Normal (or Gaussian) random number distribution
    std::normal_distribution<T> d;
};

/*! \class Mvnormdist
 * \brief Multivariate normal distribution
 * This class is independent.
 *
 * 
 * \tparam T data type one of float, double, or long double
 *
 */
template <typename T>
class Mvnormdist
{
  public:
    /*!
     * \brief constructor
     *
     * \param mean_       mean vector of size \f$n\f$
     * \param covariance_ variance-covariance matrix of size \f$n \times n\f$
     */
    Mvnormdist(EVectorX<T> const &mean_, EMatrixX<T> const &covariance_) : mean(mean_),
                                                                           covariance(covariance_),
                                                                           gen(std::random_device{}()),
                                                                           lu(covariance_)
    {
        // Computes eigenvalues and eigenvectors of selfadjoint matrices.
        Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
        transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    }

    /*!
     * \brief constructor
     * 
     * \param mean_       mean vector of size \f$n\f$
     * \param covariance_ variance-covariance matrix of size \f$n \times n\f$
     * \param n           vector size
     */
    Mvnormdist(T const *mean_, T const *covariance_, int const n) : mean(CTEMapX<T>(mean_, n, 1)),
                                                                    covariance(CTEMapX<T>(covariance_, n, n)),
                                                                    gen(std::random_device{}()),
                                                                    lu(CTEMapX<T>(covariance_, n, n))
    {
        // Computes eigenvalues and eigenvectors of selfadjoint matrices.
        Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
        transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    }

    /*!
     * \brief constructor (default mean = 0)
     *
     * \param covariance_ variance-covariance matrix of size \f$n \times n\f$
     */
    Mvnormdist(EMatrixX<T> const &covariance_) : Mvnormdist(EVectorX<T>::Zero(covariance_.rows()), covariance_) {}

    /*!
     * \brief constructor (default mean = 0)
     * 
     * \param covariance_ covariance matrix of size \f$n \times n\f$
     * \param n           vector size
     */
    Mvnormdist(T const *covariance_, int const n) : mean(EVectorX<T>::Zero(n)),
                                                    covariance(CTEMapX<T>(covariance_, n, n)),
                                                    gen(std::random_device{}()),
                                                    lu(CTEMapX<T>(covariance_, n, n))
    {
        // Computes eigenvalues and eigenvectors of selfadjoint matrices.
        Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(covariance);
        transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    }

    /*!
     * \brief constructor (default mean = 0, covariance=I)
     * 
     * \param n vector size
     */
    Mvnormdist(int const n) : mean(EVectorX<T>::Zero(n)),
                              covariance(EMatrixX<T>::Identity(n, n)),
                              transform(EMatrixX<T>::Identity(n, n)),
                              gen(std::random_device{}()),
                              lu(EMatrixX<T>::Identity(n, n)) {}

    /*!
     * \returns a vector with multivariate normal distribution
     */
    EVectorX<T> operator()()
    {
        return mean + transform * EVectorX<T>{mean.size()}.unaryExpr([&](T x) { return d(gen); });
    }

    /*!
     * \brief PDF
     * 
     * \param X vector of size \f$ n \f$
     * 
     * \returns pdf of X
     */
    T pdf(EVectorX<T> const &X)
    {
        T denom = std::pow(M_2PI, X.rows()) * lu.determinant();

        EVectorX<T> ax = X - mean;

        //Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T MDistSq = ax.transpose() * lu.inverse() * ax;

        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }

    /*!
     * \brief PDF
     * 
     * \param x vector of size \f$ n \f$
     * \param n vector size
     * 
     * \returns pdf of x
     */
    T pdf(T const *x, int const n)
    {
        CTEMapX<T, Eigen::ColMajor> X(x, n, 1);

        T denom = std::pow(M_2PI, n) * lu.determinant();

        EVectorX<T> ax = X - mean;

        //Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T MDistSq = ax.transpose() * lu.inverse() * ax;

        return std::exp(-0.5 * MDistSq) / std::sqrt(denom);
    }

    /*!
     * \brief LOGPDF
     * 
     * \param X vector of size \f$ n \f$
     * 
     * \returns logpdf of X
     */
    T lnpdf(EVectorX<T> const &X)
    {
        EVectorX<T> ax = X - mean;

        //Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T MDistSq = ax.transpose() * lu.inverse() * ax;

        return -0.5 * (MDistSq + X.rows() * M_L2PI + std::log(lu.determinant()));
    }

    /*!
     * \brief LOGPDF
     * 
     * \param x vector of size \f$ n \f$
     * \param n vector size
     * 
     * \returns logpdf of x
     */
    T lnpdf(T const *x, int const n)
    {
        CTEMapX<T, Eigen::ColMajor> X(x, n, 1);

        EVectorX<T> ax = X - mean;

        //Mahalanobis distance between \f$ X \f$ and \f$ \mu \f$
        T MDistSq = ax.transpose() * lu.inverse() * ax;

        return -0.5 * (MDistSq + n * M_L2PI + std::log(lu.determinant()));
    }

  private:
    //! Vector of size \f$n\f$
    EVectorX<T> mean;
    
    //! Variance-covariance matrix of size \f$n \times n\f$
    EMatrixX<T> covariance;
    
    //! Matrix of size \f$n \times n\f$
    EMatrixX<T> transform;
    
    //! LU decomposition of a matrix with complete pivoting
    Eigen::FullPivLU<EMatrixX<T>> lu;
    
    //! A random number engine based on Mersenne Twister algorithm
    std::mt19937 gen;
    
    //! Generates random numbers according to the Normal (or Gaussian) random number distribution
    std::normal_distribution<T> d;
};

#endif
