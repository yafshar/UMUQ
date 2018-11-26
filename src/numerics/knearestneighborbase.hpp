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
 * \tparam DataType           Data type
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
template <typename DataType, class FlannDistanceType>
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
    kNearestNeighborBase(int const ndataPoints, int const nDim, int const kNeighbors);

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
    kNearestNeighborBase(kNearestNeighborBase<DataType, FlannDistanceType> &&other);

    /*!
     * \brief Copy constructor
     * \param other kNearestNeighborBase to be copied
     */
    kNearestNeighborBase(kNearestNeighborBase<DataType, FlannDistanceType> const &other);

    /*!
     * \brief Move assignment operator
     * \param other kNearestNeighborBase to be assigned
     */
    kNearestNeighborBase<DataType, FlannDistanceType> &operator=(kNearestNeighborBase<DataType, FlannDistanceType> &&other);

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
    virtual void buildIndex(DataType *idata);

    /*!
     * \brief Construct a kd-tree index & do a knn search
     * 
     * \param idata A pointer to input data 
     * \param qdata A pointer to query data 
     */
    virtual void buildIndex(DataType *idata, DataType *qdata);

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
    inline DataType *NearestNeighborsDistances(int const &index) const;

    /*!
     * \brief Distance of a nearest neighbor of index
     * 
     * \param index Index of a point (from data points) 
     * 
     * \returns Distance of a nearest neighbor point of the index
     */
    inline DataType minDist(int const &index) const;

    /*!
     * \brief Vector of all points' distance of their nearest neighbor 
     * 
     * \returns Vector of all points' distance of their nearest neighbor 
     */
    inline DataType *minDist();

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

    /*!
     * \brief Check to see whether it requires the covariance for finding the neighbors. (Default is no)
     * 
     * \returns true If the NeighborDistance type is the Mahalanobis distance
     */
    inline bool needsCovariance() const;

    /*!
     * \brief Set the Covariance object
     *
     * \param Covariance  The covariance matrix
     */
    virtual void setCovariance(EMatrixX<DataType> const &Covariance);

    /*!
     * \brief Set the Covariance object
     *
     * \param Covariance  The covariance matrix
     */
    virtual void setCovariance(DataType const *Covariance);

    /*!
     * \brief Set the Covariance object
     *
     * \param Covariance  The covariance matrix
     */
    virtual void setCovariance(std::vector<DataType> const &Covariance);

    /*!
     * \brief Access the covariance matrix.
     *
     * \returns Constant reference to the covariance matrix.
     */
    virtual EMatrixX<DataType> const &Covariance() const;

    /*!
     * \brief Modify the covariance matrix.
     *
     * \returns Reference to the covariance matrix.
     */
    virtual EMatrixX<DataType> &Covariance();

  protected:
    /*!
     * \brief Explicitly prevent the default construct of a new kNearestNeighborBase object
     * 
     */
    kNearestNeighborBase() = delete;

  protected:
    //! Number of data points
    std::size_t nDataPoints;

    //! Number of query data points
    std::size_t nQueryDataPoints;

    //! Number of columns or dimension of each point
    std::size_t dataDimension;

    //! Number of nearest neighbors to find
    int nNearestNeighborsToFind;

    //! Pointer to keep the indices
    std::unique_ptr<int[]> indices_ptr;

    //! Pointer of distances
    std::unique_ptr<DataType[]> dists_ptr;

    //! Matrix of indices
    flann::Matrix<int> indices;

    //! Matrix of distances
    flann::Matrix<DataType> dists;

    //! Flag to check if the input data and query data are the same
    bool the_same;

    //!  Whether it requires the covariance for finding the neighbors. (Default is no)
    bool withCovariance;
};

template <typename DataType, class FlannDistanceType>
kNearestNeighborBase<DataType, FlannDistanceType>::kNearestNeighborBase(int const ndataPoints, int const nDim, int const kNeighbors) : nDataPoints(ndataPoints),
                                                                                                                                       nQueryDataPoints(ndataPoints),
                                                                                                                                       dataDimension(nDim),
                                                                                                                                       nNearestNeighborsToFind(kNeighbors + 1),
                                                                                                                                       indices_ptr(new int[ndataPoints * (kNeighbors + 1)]),
                                                                                                                                       dists_ptr(new DataType[ndataPoints * (kNeighbors + 1)]),
                                                                                                                                       indices(indices_ptr.get(), ndataPoints, (kNeighbors + 1)),
                                                                                                                                       dists(dists_ptr.get(), ndataPoints, (kNeighbors + 1)),
                                                                                                                                       the_same(true),
                                                                                                                                       withCovariance(false)
{
    if (nDataPoints < static_cast<std::size_t>(nNearestNeighborsToFind))
    {
        UMUQFAIL("Not enough points to create K nearest neighbors for each point !");
    }
}

template <typename DataType, class FlannDistanceType>
kNearestNeighborBase<DataType, FlannDistanceType>::kNearestNeighborBase(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : nDataPoints(ndataPoints),
                                                                                                                                                               nQueryDataPoints(nqueryPoints),
                                                                                                                                                               dataDimension(nDim),
                                                                                                                                                               nNearestNeighborsToFind(kNeighbors),
                                                                                                                                                               indices_ptr(new int[nqueryPoints * kNeighbors]),
                                                                                                                                                               dists_ptr(new DataType[nqueryPoints * kNeighbors]),
                                                                                                                                                               indices(indices_ptr.get(), nqueryPoints, kNeighbors),
                                                                                                                                                               dists(dists_ptr.get(), nqueryPoints, kNeighbors),
                                                                                                                                                               the_same(false),
                                                                                                                                                               withCovariance(false)
{
    if (nDataPoints < static_cast<std::size_t>(nNearestNeighborsToFind))
    {
        UMUQFAIL("Not enough points to create K nearest neighbors for each point !");
    }
}

template <typename DataType, class FlannDistanceType>
kNearestNeighborBase<DataType, FlannDistanceType>::kNearestNeighborBase(kNearestNeighborBase<DataType, FlannDistanceType> &&other) : nDataPoints(other.nDataPoints),
                                                                                                                                     nQueryDataPoints(other.nQueryDataPoints),
                                                                                                                                     dataDimension(other.dataDimension),
                                                                                                                                     nNearestNeighborsToFind(other.nNearestNeighborsToFind),
                                                                                                                                     indices_ptr(std::move(other.indices_ptr)),
                                                                                                                                     dists_ptr(std::move(other.dists_ptr)),
                                                                                                                                     indices(std::move(other.indices)),
                                                                                                                                     dists(std::move(other.dists)),
                                                                                                                                     the_same(other.the_same),
                                                                                                                                     withCovariance(other.withCovariance)
{
}

template <typename DataType, class FlannDistanceType>
kNearestNeighborBase<DataType, FlannDistanceType>::kNearestNeighborBase(kNearestNeighborBase<DataType, FlannDistanceType> const &other) : nDataPoints(other.nDataPoints),
                                                                                                                                          nQueryDataPoints(other.nQueryDataPoints),
                                                                                                                                          dataDimension(other.dataDimension),
                                                                                                                                          nNearestNeighborsToFind(other.nNearestNeighborsToFind),
                                                                                                                                          indices_ptr(new int[other.nQueryDataPoints * other.nNearestNeighborsToFind]),
                                                                                                                                          dists_ptr(new DataType[other.nQueryDataPoints * other.nNearestNeighborsToFind]),
                                                                                                                                          indices(indices_ptr.get(), other.nQueryDataPoints, other.nNearestNeighborsToFind),
                                                                                                                                          dists(dists_ptr.get(), other.nQueryDataPoints, other.nNearestNeighborsToFind),
                                                                                                                                          the_same(other.the_same),
                                                                                                                                          withCovariance(other.withCovariance)
{
    {
        int *From = other.indices_ptr.get();
        int *To = indices_ptr.get();
        std::copy(From, From + nQueryDataPoints * nNearestNeighborsToFind, To);
    }
    {
        DataType *From = other.dists_ptr.get();
        DataType *To = dists_ptr.get();
        std::copy(From, From + nQueryDataPoints * nNearestNeighborsToFind, To);
    }
}

template <typename DataType, class FlannDistanceType>
kNearestNeighborBase<DataType, FlannDistanceType> &kNearestNeighborBase<DataType, FlannDistanceType>::operator=(kNearestNeighborBase<DataType, FlannDistanceType> &&other)
{
    nDataPoints = std::move(other.nDataPoints);
    nQueryDataPoints = std::move(other.nQueryDataPoints);
    dataDimension = std::move(other.dataDimension);
    nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    indices_ptr = std::move(other.indices_ptr);
    dists_ptr = std::move(other.dists_ptr);
    indices = std::move(other.indices);
    dists = std::move(other.dists);
    the_same = std::move(other.the_same);
    withCovariance = std::move(other.withCovariance);
    return *this;
}

template <typename DataType, class FlannDistanceType>
kNearestNeighborBase<DataType, FlannDistanceType>::~kNearestNeighborBase() {}

template <typename DataType, class FlannDistanceType>
void kNearestNeighborBase<DataType, FlannDistanceType>::buildIndex(DataType *idata)
{
    flann::Matrix<DataType> dataset(idata, nDataPoints, dataDimension);

    // Construct an randomized kd-tree index using 4 kd-trees
    // For the number of parallel kd-trees to use (Good values are in the range [1..16])
    flann::Index<FlannDistanceType> index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();

    // Do a knn search, using 128 checks
    // Number of checks means: How many leafs to visit when searching
    // for neighbors (-1 for unlimited)
    index.knnSearch(dataset, indices, dists, nNearestNeighborsToFind, flann::SearchParams(128));
}

template <typename DataType, class FlannDistanceType>
void kNearestNeighborBase<DataType, FlannDistanceType>::buildIndex(DataType *idata, DataType *qdata)
{
    flann::Matrix<DataType> dataset(idata, nDataPoints, dataDimension);

    // Construct an randomized kd-tree index using 4 kd-trees
    // For the number of parallel kd-trees to use (Good values are in the range [1..16])
    flann::Index<FlannDistanceType> index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();

    flann::Matrix<DataType> query(qdata, nQueryDataPoints, dataDimension);

    // Do a knn search, using 128 checks
    // Number of checks means: How many leafs to visit when searching
    // for neighbors (-1 for unlimited)
    index.knnSearch(query, indices, dists, nNearestNeighborsToFind, flann::SearchParams(128));

    if (!checkNearestNeighbors())
    {
        UMUQWARNING("Input data & query data are the same!");
    }
}

template <typename DataType, class FlannDistanceType>
inline int *kNearestNeighborBase<DataType, FlannDistanceType>::NearestNeighbors(int const &index) const
{
    // +1 is that we do not want the index of the point itself
    return indices_ptr.get() + index * nNearestNeighborsToFind + the_same;
}

template <typename DataType, class FlannDistanceType>
inline int *kNearestNeighborBase<DataType, FlannDistanceType>::NearestNeighbors() const
{
    return indices_ptr.get();
}

template <typename DataType, class FlannDistanceType>
inline DataType *kNearestNeighborBase<DataType, FlannDistanceType>::NearestNeighborsDistances(int const &index) const
{
    // +1 is that we do not want the index of the point itself
    return dists_ptr.get() + index * nNearestNeighborsToFind + the_same;
}

template <typename DataType, class FlannDistanceType>
inline DataType kNearestNeighborBase<DataType, FlannDistanceType>::minDist(int const &index) const
{
    std::ptrdiff_t const Id = index * nNearestNeighborsToFind + the_same;
    return dists_ptr[Id];
}

template <typename DataType, class FlannDistanceType>
inline DataType *kNearestNeighborBase<DataType, FlannDistanceType>::minDist()
{
    DataType *mindists = nullptr;
    try
    {
        mindists = new DataType[nQueryDataPoints];
    }
    catch (std::bad_alloc &e)
    {
        UMUQFAILRETURNNULL("Failed to allocate memory!");
    }

    for (std::size_t i = 0; i < nQueryDataPoints; ++i)
    {
        std::ptrdiff_t const Id = i * nNearestNeighborsToFind + the_same;
        mindists[i] = dists_ptr[Id];
    }

    return mindists;
}

template <typename DataType, class FlannDistanceType>
inline int kNearestNeighborBase<DataType, FlannDistanceType>::numNearestNeighbors() const
{
    return nNearestNeighborsToFind - the_same;
}

template <typename DataType, class FlannDistanceType>
bool kNearestNeighborBase<DataType, FlannDistanceType>::checkNearestNeighbors()
{
    if (the_same)
    {
        return true;
    }
    DataType const eps = std::numeric_limits<DataType>::epsilon();
    std::size_t s(0);
    for (std::size_t i = 0; i < nQueryDataPoints; ++i)
    {
        std::ptrdiff_t const Id = i * nNearestNeighborsToFind;
        s += (dists_ptr[Id] < eps);
    }
    return (s != nQueryDataPoints);
}

template <typename DataType, class FlannDistanceType>
inline void kNearestNeighborBase<DataType, FlannDistanceType>::IndexSwap(int Indx1, int Indx2)
{
    std::swap(indices_ptr[Indx1], indices_ptr[Indx2]);
    std::swap(dists_ptr[Indx1], dists_ptr[Indx2]);
}

template <typename DataType, class FlannDistanceType>
inline int kNearestNeighborBase<DataType, FlannDistanceType>::numInputdata() const { return nDataPoints; }

template <typename DataType, class FlannDistanceType>
inline int kNearestNeighborBase<DataType, FlannDistanceType>::numQuerydata() const { return nQueryDataPoints; }

template <typename DataType, class FlannDistanceType>
inline bool kNearestNeighborBase<DataType, FlannDistanceType>::needsCovariance() const { return withCovariance; }

template <typename DataType, class FlannDistanceType>
void kNearestNeighborBase<DataType, FlannDistanceType>::setCovariance(EMatrixX<DataType> const &Covariance) {}

template <typename DataType, class FlannDistanceType>
void kNearestNeighborBase<DataType, FlannDistanceType>::setCovariance(DataType const *Covariance) {}

template <typename DataType, class FlannDistanceType>
void kNearestNeighborBase<DataType, FlannDistanceType>::setCovariance(std::vector<DataType> const &Covariance) {}

template <typename DataType, class FlannDistanceType>
EMatrixX<DataType> const &kNearestNeighborBase<DataType, FlannDistanceType>::Covariance() const {}

template <typename DataType, class FlannDistanceType>
EMatrixX<DataType> &kNearestNeighborBase<DataType, FlannDistanceType>::Covariance() {}

} // namespace umuq

#endif // UMUQ_KNEARESTNEIGHBORBASE
