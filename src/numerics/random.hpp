#ifndef UMHBM_RANDOM_H
#define UMHBM_RANDOM_H

#include <memory>
#include <random>
#include "../../external/torc/include/torc.h"
#include "eigenmatrix.hpp"
#include "saruprng.hpp"

/*! 
 * 
 */
std::uint32_t **rseed = nullptr;




/*! 
 * 32-bit Mersenne Twister by Matsumoto and Nishimura, 1998
 */
static std::mt19937 NumberGenerator;

/*! 
 * produces real values on a standard normal (Gaussian) distribution
 */
static std::normal_distribution<> normalDist;

/*! 
 * C++ Saru PRNG
 */
Saru saru;

/*! 
 * Sets the current state of the engine 
 */
void MTinit(size_t iseed = std::random_device{}())
{
    int local_workers = torc_i_num_workers();
    int torc_node_id = torc_node_id();

    rseed = new std::uint32_t *[local_workers];
    for (int i = 0; i < local_workers; i++)
    {
        rseed[i] = new std::uint32_t[std::mt19937::state_size];
        for (int j = 0; j < std::mt19937::state_size; j++)
        {
            rseed[i][j] = (std::uint32_t)(iseed + i + local_workers * torc_node_id);
        }
    }

    // Seed the engine with unsigned ints
    std::seed_seq sseq(r, r + local_workers);

    NumberGenerator.seed(sseq);
}

void spmd_MTinit()
{
    for (int i = 0; i < torc_num_nodes(); i++)
    {
        torc_create_ex(i * torc_i_num_workers(), 1, (void (*)())MTinit, 0);
    }
    torc_waitall();
}

/*! 
 * Sets the current state of the engine 
 */
void Saruinit(size_t iseed = std::random_device{}())
{
    int local_workers = torc_i_num_workers();
    int torc_node_id = torc_node_id();

    std::uint32_t r[local_workers];
    for (int i = 0; i < local_workers; i++)
    {
        r[i] = (std::uint32_t)(iseed + i + local_workers * torc_node_id);
    }

    // Seed the engine with unsigned ints
    std::seed_seq sseq(r, r + local_workers);

    NumberGenerator.seed(sseq);
}

/*!
 * \brief Multivariate normal distribution
 * \tparam TM the type of the Matrix 
 * \tparam TV the type of the Vector
 */
template <typename TM, typename TV>
struct mvnormdist
{
    mvnormdist(TM const &covariance) : mvnormdist(TV::Zero(covariance.rows()), covariance) {}
    mvnormdist(TV const &mean, TM const &covariance) : mean(mean)
    {
        // Computes eigenvalues and eigenvectors of selfadjoint matrices.
        Eigen::SelfAdjointEigenSolver<TV> es(covariance);
        transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    }

    TV mean;
    TM transform;

    TV operator()() const
    {
        NumberGenerator{std::random_device{}()};
        return mean + transform * TV{mean.size()}.unaryExpr([&](TM::Scalar x) { return dist(NumberGenerator); });
    }
};

/*!
 * \brief Generates random numbers according to the Normal (or Gaussian) random number distribution
 */
template <typename T>
struct normrnd
{
    normrnd(T mean = 0, T stddev = 1)
    {
        // values near the mean are the most likely
        // standard deviation affects the dispersion of generated values from the mean
        std::normal_distribution<T> d{mean, stddev};
    }

    T d;

    /*!
     * \brief access element at provided index 
     * @param id
     * @return element @(id)
     */
    T operator()() const
    {
        return d(NumberGenerator);
    }
}

/* uniform (flat) distribution random number between a and b */
double
uniformrand(double a, double b)
{
    double res;

    int me = torc_i_worker_id();
    res = gsl_ran_flat(r[me], a, b);

    return res;

    /*    return gsl_ran_flat(r, a, b);*/
}

void multinomialrand(size_t K, unsigned int N, double q[], unsigned int nn[])
{
    int me = torc_i_worker_id();
    gsl_ran_multinomial(r[me], K, N, q, nn);

    return;
}

#endif
