#ifndef UMHBM_PSRANDOM_H
#define UMHBM_PSRANDOM_H

#include "saruprng.hpp"
#include "factorial.hpp"

/*! \class psrandom
  *
  */
struct psrandom
{
    /*!
     * \brief Default constructor
     */
    psrandom(){};

    /*!
     * \brief constructor
     * 
     * \param seed Input seed for random number initialization 
     */
    psrandom(size_t const &iseed_);

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
     * \brief Init task on each node to set the current state of the engine for each thread
     */
    static void init_Task();

    /*!
     * \returns \a true when sets the current state of the engine successfully
     * 
     * \brief Sets the current state of the engine
     */
    bool init();

    /*!
     * \returns Uniform random number between [a..b)
     * 
     * \brief Uniform random number between [a..b)
     *
     * \tparam T data type one of float, double
     *
     * Advance the PRNG state by 1, and output a T precision [a..b) number (default a = 0, b = 1)
     */
    template <typename T>
    inline T unirnd(T const a = 0, T const b = 1)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " The Factorial of type " << typeid(T).name() << " is not implemented !" << std::endl;
    }

    /*!
     * \returns a uniform random number of a double precision [0..1) floating point 
     * 
     * \brief Advance state by 1, and output a double precision [0..1) floating point
     * 
     * Reference:
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
     */
    inline double d() { return saru[0].d(); }

    /*!
     * \returns a uniform random number of a single precision [0..1) floating point
     * 
     * \Advance state by 1, and output a single precision [0..1) floating point
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
    void shuffle(int *idata, int const nSize)
    {
        //Get the thread ID
        int const me = torc_i_worker_id();

        for (int i = nSize - 1; i > 0; --i)
        {
            unsigned int const idx = saru[me].u32(i);
            std::swap(idata[i], idata[idx]);
        }
    }

    /*!
     * RNG seed
     */
    static size_t iseed;

    /*! 
     * 32-bit Mersenne Twister by Matsumoto and Nishimura, 1998
     */
    static std::mt19937 *NumberGenerator;

    /*! 
     * C++ Saru PRNG
     */
    static Saru *saru;
};

size_t psrandom::iseed = 0;
std::mt19937 *psrandom::NumberGenerator = nullptr;
Saru *psrandom::saru = nullptr;

/*!
 * \brief constructor
 * 
 * \param seed input seed for random number initialization 
 */
psrandom::psrandom(size_t const &iseed_)
{
    psrandom::iseed = iseed_;

    try
    {
        //Number of local workers
        int const nlocalworkers = torc_i_num_workers();

        psrandom::NumberGenerator = new std::mt19937[nlocalworkers];
        psrandom::saru = new Saru[nlocalworkers];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    };
}

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

    if (psrandom::iseed == 0)
    {
        psrandom::iseed = std::random_device{}();
    }

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
 * \brief Uniform random number between [a..b)
 *
 * \tparam T data type one of float, double
 *
 * Advance the PRNG state by 1, and output a T precision [a..b) number (default a = 0, b = 1)
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
 * This is a partial specialization to make a special case for float precision uniform random number
 */
template <>
inline float psrandom::unirnd<float>(float a, float b)
{
    //Get the thread ID
    int const me = torc_i_worker_id();
    return saru[me].f(a, b);
}

/*! 
 * multinomial distribution
 *
 * This contains minor modifications and adaptation to the original 
 * source code made available under the following license:
 *
 * \verbatim
 * Copyright (C) 2002 Gavin E. Crooks <gec@compbio.berkeley.edu>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 * \endverbatim
 */

/*!   
 * \brief The multinomial distribution
 * 
 * This is based on psrandom object seeded engine.  
 * So to use it there should be an instance of psrandom object.
 * 
 * \tparam T data 
 * \param    K  K possible mutually exclusive outcomes 
 * \param    N  N independent trials
 * \param    p  Corresponding probabilities \f$ p_1, \cdots, p_k \f$
 * \param    n  A random sample \f$n_1, \cdots, n_K\f$ from the multinomial distribution
 * 
 * 
 * The multinomial distribution has the form
 * \f[
 *     p(n_1, n_2, ... n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \codts p_K^{n_K}
 * \f] 
 *
 * where \f$ n_1, n_2, \cdots n_K \f$ are nonnegative integers, \f$ sum_{k=1,K} {n_k = N}\f$,
 * and \f$p = \left(p_1, p_2, \cdots, p_K\right)\f$ is a probability distribution. 
 *
 * Random variates are generated using the conditional binomial method.
 * This scales well with N and does not require a setup step.
 *   
 *  Ref: 
 *  C.S. David, The computer generation of multinomial random variates,
 *  Comp. Stat. Data Anal. 16 (1993) 205-217
 */
template <typename T = double>
bool multinomial(size_t const K, unsigned int const N, T const *p, unsigned int *n)
{
    if (psrandom::iseed == 0)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "You should create an instance of a psrandom object before using this class!" << std::endl;
        return false;
    }

    //Get the thread ID
    int const me = torc_i_worker_id();

    T const norm = std::accumulate(p, p + K, 0);
    T sum_p = 0.0;
    unsigned int sum_n = 0;
    for (size_t i = 0; i < K; i++)
    {
        if (p[i] > 0.0)
        {
            std::binomial_distribution<> d(N - sum_n, p[i] / (norm - sum_p));
            n[i] = d(psrandom::NumberGenerator[me]);
        }
        else
        {
            n[i] = 0;
        }
        sum_p += p[i];
        sum_n += n[i];
    }
    return true;
}

/*!   
 * \brief The Multinomial distribution
 * 
 * This is independent. 
 * 
 * \tparam T data 
 * \param    K  K possible mutually exclusive outcomes 
 * \param    N  N independent trials
 * \param    p  Corresponding probabilities \f$ p_1, \cdots, p_k \f$
 * \param    n  A random sample \f$n_1, \cdots, n_K\f$ from the multinomial distribution
 * 
 * 
 * The multinomial distribution has the form
 * \f[
 *     p(n_1, n_2, ... n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \codts p_K^{n_K}
 * \f] 
 *
 * where \f$ n_1, n_2, \cdots n_K \f$ are nonnegative integers, \f$ sum_{k=1,K} {n_k = N}\f$,
 * and \f$p = \left(p_1, p_2, \cdots, p_K\right)\f$ is a probability distribution. 
 *
 * Random variates are generated using the conditional binomial method.
 * This scales well with N and does not require a setup step.
 *   
 *  Ref: 
 *  C.S. David, The computer generation of multinomial random variates,
 *  Comp. Stat. Data Anal. 16 (1993) 205-217
 */
template <typename T = double>
bool Multinomial(size_t const K, unsigned int const N, T const *p, unsigned int *n)
{
    std::mt19937 gen(std::random_device{}());

    T const norm = std::accumulate(p, p + K, 0);
    T sum_p = 0.0;
    unsigned int sum_n = 0;
    for (size_t i = 0; i < K; i++)
    {
        if (p[i] > 0.0)
        {
            std::binomial_distribution<> d(N - sum_n, p[i] / (norm - sum_p));
            n[i] = d(gen);
        }
        else
        {
            n[i] = 0;
        }
        sum_p += p[i];
        sum_n += n[i];
    }
    return true;
}

/*!
 * \brief Computes the the logarithm of the probability \f$p(n_1, n_2, \cdots, n_K)\f$
 * 
 * This is based on psrandom object seeded engine.  
 * So to use it there should be an instance of psrandom object.
 * 
 * This function computes the logarithm of the probability \f$p(n_1, n_2, \cdots, n_K)\f$ of sampling \f$n[K]\f$ 
 * from a multinomial distribution with parameters \f$p[K]\f$, using the formula given above
 * 
 * \tparam T data type one of float, double
 * \param  n A random sample \f$n_1, \cdots, n_K\f$ from the multinomial distribution
 * 
 * \returns the logarithm of the probability \f$p(n_1, n_2, \cdots, n_K)\f$ of sampling \f$n[K]\f$ 
 */
template <typename T = double>
T multinomial_lnpdf(size_t const K, T const *p, unsigned int const *n)
{
    unsigned int N = std::accumulate(n, n + K, 0);
    T const norm = std::accumulate(p, p + K, 0);
    T log_pdf = factorial<T>(N);

    for (size_t i = 0; i < K; i++)
    {
        if (n[i] > 0)
        {
            log_pdf += std::log(p[i] / norm) * n[i] - factorial<T>(n[i]);
        }
    }

    return log_pdf;
}

/*!
 * \brief computes the probability \f$p(n_1, n_2, \cdots, n_K)\f$
 * 
 * This is based on psrandom object seeded engine.  
 * So to use it there should be an instance of psrandom object.
 * 
 * This function computes the probability \f$p(n_1, n_2, \cdots, n_K)\f$ of sampling \f$n[K]\f$ 
 * from a multinomial distribution with parameters \f$p[K]\f$, using the formula given above
 * 
 * \tparam T data type one of float, double
 * \param  n A random sample \f$n_1, \cdots, n_K\f$ from the multinomial distribution
 * 
 * \returns the probability \f$p(n_1, n_2, \cdots, n_K)\f$ of sampling \f$n[K]\f$
 */
template <typename T = double>
T multinomial_pdf(size_t const K, T const *p, unsigned int const *n)
{
    unsigned int N = std::accumulate(n, n + K, 0);
    T const norm = std::accumulate(p, p + K, 0);
    T log_pdf = factorial<T>(N);

    for (size_t i = 0; i < K; i++)
    {
        if (n[i] > 0)
        {
            log_pdf += std::log(p[i] / norm) * n[i] - factorial<T>(n[i]);
        }
    }

    return std::exp(log_pdf);
}

/*!
 * \brief Generates random numbers according to the Normal (or Gaussian) random number distribution
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
            std::cerr << "You should create an instance of a psrandom object before using this class!" << std::endl;
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
    std::normal_distribution<T> d;
};

/*!
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
    std::mt19937 gen;
    std::normal_distribution<T> d;
};

/*!
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
            std::cerr << "You should create an instance of a psrandom object before using this class!" << std::endl;
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
    std::lognormal_distribution<T> d;
};

/*!
 * \brief Generates random numbers x > 0 according to the lognormal_distribution
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
    std::mt19937 gen;
    std::lognormal_distribution<T> d;
};

/*! \class mvnormdist
 * \brief Multivariate normal distribution
 * This class is based on psrandom object seeded engine. 
 * So to use this object there should be an instance of psrandom object.
 *
 * \tparam TM the type of the Matrix
 * \tparam TV the type of the Vector
 *
 * \tparam T data type one of float, double, or long double
 *
 */
template <typename TM, typename TV>
class mvnormdist
{
  public:
    /*!
     * \brief constructor (default mean = 0)
     *
     * \param covariance covariance Matrix
     */
    mvnormdist(TM const &covariance) : mvnormdist(TV::Zero(covariance.rows()), covariance) {}

    /*!
     * \brief constructor
     *
     * \param mean mean vector
     * \param covariance covariance Matrix
     */
    mvnormdist(TV const &mean, TM const &covariance) : mean(mean)
    {
        if (psrandom::iseed == 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "You should create an instance of a psrandom object before using this class!" << std::endl;
        }

        // Computes eigenvalues and eigenvectors of selfadjoint matrices.
        Eigen::SelfAdjointEigenSolver<TV> es(covariance);
        transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    }

    /*!
     * \returns a vector with multivariate normal distribution
     */
    TV operator()() const
    {
        int const me = torc_i_worker_id();
        return mean + transform * TV{mean.size()}.unaryExpr([&](T x) { return d(psrandom::NumberGenerator[me]); });
    }

  private:
    TV mean;
    TM transform;
    typedef typename TM::Scalar T;
    std::normal_distribution<T> d;
};

/*! \class Mvnormdist
 * \brief Multivariate normal distribution
 * This class is independent.
 *
 * \tparam TM the type of the Matrix
 * \tparam TV the type of the Vector
 *
 * \tparam T data type one of float, double, or long double
 *
 */
template <typename TM, typename TV>
class Mvnormdist
{
  public:
    /*!
     * \brief constructor (default mean = 0)
     *
     * \param covariance covariance Matrix
     */
    Mvnormdist(TM const &covariance) : Mvnormdist(TV::Zero(covariance.rows()), covariance) {}

    /*!
     * \brief constructor
     *
     * \param mean mean vector
     * \param covariance covariance Matrix
     */
    Mvnormdist(TV const &mean, TM const &covariance) : mean(mean), gen(std::random_device{}())
    {
        // Computes eigenvalues and eigenvectors of selfadjoint matrices.
        Eigen::SelfAdjointEigenSolver<TV> es(covariance);
        transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    }

    /*!
     * \returns a vector with multivariate normal distribution
     */
    TV operator()() const
    {
        return mean + transform * TV{mean.size()}.unaryExpr([&](T x) { return d(gen); });
    }

  private:
    TV mean;
    TM transform;
    typedef typename TM::Scalar T;
    std::mt19937 gen;
    std::normal_distribution<T> d;
};

#endif
