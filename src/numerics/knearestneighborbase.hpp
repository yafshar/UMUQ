#ifndef UMUQ_KNEARESTNEIGHBORBASE_H
#define UMUQ_KNEARESTNEIGHBORBASE_H

#include "eigenlib.hpp"

/*!
 * \ingroup Numerics_Module
 * 
 * FLANN is a library for performing fast approximate nearest neighbor searches in high dimensional spaces. 
 * It contains a collection of algorithms we found to work best for nearest neighbor search and a system 
 * for automatically choosing the best algorithm and optimum parameters depending on the dataset.
 */
#include <flann/flann.hpp>

namespace umuq
{

/*! \class kNearestNeighborBase
 * \ingroup Numerics_Module
 * 
 * \brief Finding K nearest neighbors in high dimensional spaces
 * 
 * \tparam T                  Data type
 * \tparam FlannDistanceType  Distance type from flann library for computing the distances to the nearest neighbors
 * 
 * Distance type for computing the distances to the nearest neighbors include below distance types from flann library:
 * - \b flann::L2                        Squared Euclidean distance functor, optimized version.
 * - \b flann::L2_Simple                 Squared Euclidean distance functor.
 * - \b flann::L1                        Manhattan distance functor, optimized version.
 * - \b flann::MinkowskiDistance         The Minkowsky \f$ (L_p) (x,y) = \left( \sum_{i = 1}^{n} | x_i - y_i |^p \right)^{\frac{1}{p}} \f$ distance between two vectors \f$ x,~\text{and}~y. \f$
 * - \b flann::MaxDistance               The Maximum distance.
 * - \b flann::HistIntersectionDistance  The histogram intersection distance.
 * - \b flann::HellingerDistance         The Hellinger distance, quantify the similarity between two probability distributions.
 * - \b flann::ChiSquareDistance         The distance between two histograms.
 * - \b flann::KL_Divergence             The Kullback-Leibler divergence.
 * - \b flann::Hamming                   Hamming distance functor.
 * - \b flann::HammingLUT                Hamming distance functor - counts the bit differences between two strings - 
 *                                       useful for the Brief descriptor bit count of A exclusive \c XOR'ed with B.
 * - \b flann::HammingPopcnt             Hamming distance functor (pop count between two binary vectors, i.e. xor them 
 *                                       and count the number of bits set).
 */
template <typename T, class FlannDistanceType>
class kNearestNeighborBase
{
  public:
    /*!
     * \brief Construct a new kNearestNeighborBase object
     * 
     * \param ndataPoints  Number of data points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighborBase(int const ndataPoints, int const nDim, int const nNearestNeighbors);

    /*!
     * \brief Construct a new kNearestNeighborBase object
     * 
     * \param ndataPoints  Number of data points
     * \param nqueryPoints Number of query points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighborBase(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Move constructor
     * \param other kNearestNeighborBase to be moved
     */
    kNearestNeighborBase(kNearestNeighborBase<T, FlannDistanceType> &&other);

    /*!
     * \brief Copy constructor
     * \param other kNearestNeighborBase to be copied
     */
    kNearestNeighborBase(kNearestNeighborBase<T, FlannDistanceType> const &other);

    /*!
     * \brief Move assignment operator
     * \param other kNearestNeighborBase to be assigned
     */
    kNearestNeighborBase<T, FlannDistanceType> &operator=(kNearestNeighborBase<T, FlannDistanceType> &&other);

    /*!
     * \brief Default destructor
     *
     */
    ~kNearestNeighborBase();

    /*!
     * \brief Construct a kd-tree index & do a knn search
     * 
     * \param idata A pointer to input data 
     */
    virtual void buildIndex(T *idata);

    /*!
     * \brief Construct a kd-tree index & do a knn search
     * 
     * \param idata A pointer to input data 
     * \param qdata A pointer to query data 
     */
    virtual void buildIndex(T *idata, T *qdata);

    /*!
     * \brief A pointer to nearest neighbors indices
     * 
     * \param index Index of a point (from data points) to get its neighbors
     * 
     * \returns A (pointer to a) row of the nearest neighbors indices.
     */
    inline int *NearestNeighbors(int const &index) const;

    /*!
     * \brief A pointer to all points nearest neighbors indices
     * 
     * The function returns a pointer of \c size(nPoints * (kNeighbors+1)) all points neighbors.<br>
     * The returned pointer looks like below:<br>
     * \verbatim
     *    0                1      .     kNeighbors
     *   ---------------------------------
     *  | 0               0_1     .     0_k
     *  | 1               1_1     .     1_k
     *  | .
     *  | nPoints-1        .      .     (nPoints-1)_k
     * \endverbatim
     * 
     * 
     * Each row has the size of n which is the number of k neighbors + 1 <br>
     * and it has \c nPoints rows. <br>
     * The first column is the indices of points themselves in case of input data and query data are the same.
     * 
     * \returns All points nearest neighbors indices (in row order).
     */
    inline int *NearestNeighbors() const;

    /*!
     * \brief A pointer to nearest neighbors distances from the point index
     * 
     * \param index Index of a point (from data points) 
     * 
     * \returns A pointer to nearest neighbors distances from the point index
     */
    inline T *NearestNeighborsDistances(int const &index) const;

    /*!
     * \brief Distance of a nearest neighbor of index
     * 
     * \param index Index of a point (from data points) 
     * 
     * \returns Distance of a nearest neighbor point of the index
     */
    inline T minDist(int const &index) const;

    /*!
     * \brief Vector of all points' distance of their nearest neighbor 
     * 
     * \returns Vector of all points' distance of their nearest neighbor 
     */
    inline T *minDist();

    /*!
     * \brief Number of each point nearest neighbors
     * 
     * \returns number of nearest neighbors
     */
    inline int numNearestNeighbors() const;

    /*!
     * \brief Function to check that we do not compute the nearest neighbors of a 
     * point from itself, in case of different input data and query data
     * 
     * \returns true If input points and query points are used correctly
     */
    bool checkNearestNeighbors();

    /*!
     * \brief swap two indexes values
     */
    inline void IndexSwap(int Indx1, int Indx2);

    /*!
     * \returns number of input data points
     */
    inline int numInputdata() const;

    /*!
     * \returns number of query data points
     */
    inline int numQuerydata() const;

  protected:
    /*!
     * \brief Explicitly prevent the default construct of a new kNearestNeighborBase object
     * 
     */
    kNearestNeighborBase() = delete;

  protected:
    //! Number of data points
    std::size_t drows;

    //! Number of query data points
    std::size_t qrows;

    //! Number of columns
    std::size_t cols;

    //! Number of nearest neighbors to find
    int nn;

    //! Pointer to keep the indices
    std::unique_ptr<int[]> indices_ptr;

    //! Pointer of distances
    std::unique_ptr<T[]> dists_ptr;

    //! Matrix of indices
    flann::Matrix<int> indices;

    //! Matrix of distances
    flann::Matrix<T> dists;

    //! Flag to check if the input data and query data are the same
    bool the_same;

    //!  Whether it requires the covariance for finding the neighbors. (Default is no)
    bool withCovariance;
};

template <typename T, class FlannDistanceType>
kNearestNeighborBase<T, FlannDistanceType>::kNearestNeighborBase(int const ndataPoints, int const nDim, int const kNeighbors) : drows(ndataPoints),
                                                                                                                                qrows(ndataPoints),
                                                                                                                                cols(nDim),
                                                                                                                                nn(kNeighbors + 1),
                                                                                                                                indices_ptr(new int[ndataPoints * (kNeighbors + 1)]),
                                                                                                                                dists_ptr(new T[ndataPoints * (kNeighbors + 1)]),
                                                                                                                                indices(indices_ptr.get(), ndataPoints, (kNeighbors + 1)),
                                                                                                                                dists(dists_ptr.get(), ndataPoints, (kNeighbors + 1)),
                                                                                                                                the_same(true),
                                                                                                                                withCovariance(false)
{
    if (drows < static_cast<std::size_t>(nn))
    {
        UMUQFAIL("Not enough points to create K nearest neighbors for each point !");
    }
}

template <typename T, class FlannDistanceType>
kNearestNeighborBase<T, FlannDistanceType>::kNearestNeighborBase(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : drows(ndataPoints),
                                                                                                                                                        qrows(nqueryPoints),
                                                                                                                                                        cols(nDim),
                                                                                                                                                        nn(kNeighbors),
                                                                                                                                                        indices_ptr(new int[nqueryPoints * kNeighbors]),
                                                                                                                                                        dists_ptr(new T[nqueryPoints * kNeighbors]),
                                                                                                                                                        indices(indices_ptr.get(), nqueryPoints, kNeighbors),
                                                                                                                                                        dists(dists_ptr.get(), nqueryPoints, kNeighbors),
                                                                                                                                                        the_same(false),
                                                                                                                                                        withCovariance(false)
{
    if (drows < static_cast<std::size_t>(nn))
    {
        UMUQFAIL("Not enough points to create K nearest neighbors for each point !");
    }
}

template <typename T, class FlannDistanceType>
kNearestNeighborBase<T, FlannDistanceType>::kNearestNeighborBase(kNearestNeighborBase<T, FlannDistanceType> &&other) : drows(other.drows),
                                                                                                                       qrows(other.qrows),
                                                                                                                       cols(other.cols),
                                                                                                                       nn(other.nn),
                                                                                                                       indices_ptr(std::move(other.indices_ptr)),
                                                                                                                       dists_ptr(std::move(other.dists_ptr)),
                                                                                                                       indices(std::move(other.indices)),
                                                                                                                       dists(std::move(other.dists)),
                                                                                                                       the_same(other.the_same),
                                                                                                                       withCovariance(other.withCovariance)
{
}

template <typename T, class FlannDistanceType>
kNearestNeighborBase<T, FlannDistanceType>::kNearestNeighborBase(kNearestNeighborBase<T, FlannDistanceType> const &other) : drows(other.drows),
                                                                                                                            qrows(other.qrows),
                                                                                                                            cols(other.cols),
                                                                                                                            nn(other.nn),
                                                                                                                            indices_ptr(new int[other.qrows * other.nn]),
                                                                                                                            dists_ptr(new T[other.qrows * other.nn]),
                                                                                                                            indices(indices_ptr.get(), other.qrows, other.nn),
                                                                                                                            dists(dists_ptr.get(), other.qrows, other.nn),
                                                                                                                            the_same(other.the_same),
                                                                                                                            withCovariance(other.withCovariance)
{
    {
        int *From = other.indices_ptr.get();
        int *To = indices_ptr.get();
        std::copy(From, From + qrows * nn, To);
    }
    {
        T *From = other.dists_ptr.get();
        T *To = dists_ptr.get();
        std::copy(From, From + qrows * nn, To);
    }
}

template <typename T, class FlannDistanceType>
kNearestNeighborBase<T, FlannDistanceType> &kNearestNeighborBase<T, FlannDistanceType>::operator=(kNearestNeighborBase<T, FlannDistanceType> &&other)
{
    drows = std::move(other.drows);
    qrows = std::move(other.qrows);
    cols = std::move(other.cols);
    nn = std::move(other.nn);
    indices_ptr = std::move(other.indices_ptr);
    dists_ptr = std::move(other.dists_ptr);
    indices = std::move(other.indices);
    dists = std::move(other.dists);
    the_same = std::move(other.the_same);
    withCovariance = std::move(other.withCovariance);
    return *this;
}

template <typename T, class FlannDistanceType>
kNearestNeighborBase<T, FlannDistanceType>::~kNearestNeighborBase() {}

template <typename T, class FlannDistanceType>
void kNearestNeighborBase<T, FlannDistanceType>::buildIndex(T *idata)
{
    flann::Matrix<T> dataset(idata, drows, cols);

    // Construct an randomized kd-tree index using 4 kd-trees
    // For the number of parallel kd-trees to use (Good values are in the range [1..16])
    flann::Index<FlannDistanceType> index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();

    // Do a knn search, using 128 checks
    // Number of checks means: How many leafs to visit when searching
    // for neighbors (-1 for unlimited)
    index.knnSearch(dataset, indices, dists, nn, flann::SearchParams(128));
}

template <typename T, class FlannDistanceType>
void kNearestNeighborBase<T, FlannDistanceType>::buildIndex(T *idata, T *qdata)
{
    flann::Matrix<T> dataset(idata, drows, cols);

    // Construct an randomized kd-tree index using 4 kd-trees
    // For the number of parallel kd-trees to use (Good values are in the range [1..16])
    flann::Index<FlannDistanceType> index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();

    flann::Matrix<T> query(qdata, qrows, cols);

    // Do a knn search, using 128 checks
    // Number of checks means: How many leafs to visit when searching
    // for neighbors (-1 for unlimited)
    index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

    if (!checkNearestNeighbors())
    {
        UMUQWARNING("Input data & query data are the same!");
    }
}

template <typename T, class FlannDistanceType>
inline int *kNearestNeighborBase<T, FlannDistanceType>::NearestNeighbors(int const &index) const
{
    // +1 is that we do not want the index of the point itself
    return indices_ptr.get() + index * nn + the_same;
}

template <typename T, class FlannDistanceType>
inline int *kNearestNeighborBase<T, FlannDistanceType>::NearestNeighbors() const
{
    return indices_ptr.get();
}

template <typename T, class FlannDistanceType>
inline T *kNearestNeighborBase<T, FlannDistanceType>::NearestNeighborsDistances(int const &index) const
{
    // +1 is that we do not want the index of the point itself
    return dists_ptr.get() + index * nn + the_same;
}

template <typename T, class FlannDistanceType>
inline T kNearestNeighborBase<T, FlannDistanceType>::minDist(int const &index) const
{
    std::ptrdiff_t const Id = index * nn + the_same;
    return dists_ptr[Id];
}

template <typename T, class FlannDistanceType>
inline T *kNearestNeighborBase<T, FlannDistanceType>::minDist()
{
    T *mindists = nullptr;
    try
    {
        mindists = new T[qrows];
    }
    catch (std::bad_alloc &e)
    {
        UMUQFAILRETURNNULL("Failed to allocate memory!");
    }

    for (std::size_t i = 0; i < qrows; ++i)
    {
        std::ptrdiff_t const Id = i * nn + the_same;
        mindists[i] = dists_ptr[Id];
    }

    return mindists;
}

template <typename T, class FlannDistanceType>
inline int kNearestNeighborBase<T, FlannDistanceType>::numNearestNeighbors() const
{
    return nn - the_same;
}

template <typename T, class FlannDistanceType>
bool kNearestNeighborBase<T, FlannDistanceType>::checkNearestNeighbors()
{
    if (the_same)
    {
        return true;
    }
    T const eps = std::numeric_limits<T>::epsilon();
    std::size_t s(0);
    for (std::size_t i = 0; i < qrows; ++i)
    {
        std::ptrdiff_t const Id = i * nn;
        s += (dists_ptr[Id] < eps);
    }
    return (s != qrows);
}

template <typename T, class FlannDistanceType>
inline void kNearestNeighborBase<T, FlannDistanceType>::IndexSwap(int Indx1, int Indx2)
{
    std::swap(indices_ptr[Indx1], indices_ptr[Indx2]);
    std::swap(dists_ptr[Indx1], dists_ptr[Indx2]);
}

template <typename T, class FlannDistanceType>
inline int kNearestNeighborBase<T, FlannDistanceType>::numInputdata() const
{
    return drows;
}

template <typename T, class FlannDistanceType>
inline int kNearestNeighborBase<T, FlannDistanceType>::numQuerydata() const
{
    return qrows;
}

} // namespace umuq

#endif // UMUQ_KNEARESTNEIGHBORBASE
