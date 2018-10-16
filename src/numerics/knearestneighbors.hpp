#ifndef UMUQ_KNEARESTNEIGHBORS_H
#define UMUQ_KNEARESTNEIGHBORS_H

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

/*! \class kNearestNeighbor
 * \ingroup Numerics_Module
 * 
 * \brief Finding K nearest neighbors in high dimensional spaces
 * 
 * \tparam T         Data type
 * \tparam Distance  Distance type for computing the distances to the nearest neighbors
 *                   (Default is a specialized class - \b kNearestNeighbor<T> with L2 distance)
 * 
 * Distance type for computing the distances to the nearest neighbors include below distance types:
 * - \b EUCLIDEAN      Squared Euclidean distance functor, optimized version 
 * - \b L2             Squared Euclidean distance functor, optimized version 
 * - \b MANHATTAN      Manhattan distance functor, optimized version
 * - \b L1             Manhattan distance functor, optimized version
 * - \b L2_SIMPLE      Squared Euclidean distance functor
 * - \b MINKOWSKI      The Minkowsky \f$ (L_p) (x,y) = \left( \sum_{i = 1}^{n} | x_i - y_i |^p \right)^{\frac{1}{p}} \f$ distance between two vectors \f$ x,~\text{and}~y. \f$
 * - \b MAX
 * - \b HIST_INTERSECT
 * - \b HELLINGER      The Hellinger distance, quantify the similarity between two probability distributions.
 * - \b CHI_SQUARE     The distance between two histograms
 * - \b KULLBACK_LEIBLER
 * - \b HAMMING
 * - \b HAMMING_LUT    Hamming distance functor - counts the bit differences between two strings - 
 *                     useful for the Brief descriptor bit count of A exclusive XOR'ed with B
 * - \b HAMMING_POPCNT Hamming distance functor (pop count between two binary vectors, i.e. xor them 
 *                     and count the number of bits set)
 */
template <typename T, class Distance>
class kNearestNeighbor
{
  public:
    /*!
     * \brief Construct a new kNearestNeighbor object
     * 
     * \param ndataPoints  Number of data points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nDim, int const nNearestNeighbors);

    /*!
     * \brief Construct a new kNearestNeighbor object
     * 
     * \param ndataPoints  Number of data points
     * \param nqueryPoints Number of query points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Move constructor
     * \param other kNearestNeighbor to be moved
     */
    kNearestNeighbor(kNearestNeighbor<T, Distance> &&other);

    /*!
     * \brief Copy constructor
     * \param other kNearestNeighbor to be copied
     */
    kNearestNeighbor(kNearestNeighbor<T, Distance> const &other);

    /*!
     * \brief Move assignment operator
     * \param other kNearestNeighbor to be assigned
     */
    kNearestNeighbor<T, Distance> &operator=(kNearestNeighbor<T, Distance> &&other);

    /*!
     * \brief Default destructor
     *
     */
    ~kNearestNeighbor();

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
};

template <typename T, class Distance>
kNearestNeighbor<T, Distance>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : drows(ndataPoints),
                                                                                                               qrows(ndataPoints),
                                                                                                               cols(nDim),
                                                                                                               nn(kNeighbors + 1),
                                                                                                               indices_ptr(new int[ndataPoints * (kNeighbors + 1)]),
                                                                                                               dists_ptr(new T[ndataPoints * (kNeighbors + 1)]),
                                                                                                               indices(indices_ptr.get(), ndataPoints, (kNeighbors + 1)),
                                                                                                               dists(dists_ptr.get(), ndataPoints, (kNeighbors + 1)),
                                                                                                               the_same(true)
{
    if (drows < static_cast<std::size_t>(nn))
    {
        UMUQFAIL("Not enough points to create K nearest neighbors for each point !");
    }
}

template <typename T, class Distance>
kNearestNeighbor<T, Distance>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : drows(ndataPoints),
                                                                                                                                       qrows(nqueryPoints),
                                                                                                                                       cols(nDim),
                                                                                                                                       nn(kNeighbors),
                                                                                                                                       indices_ptr(new int[nqueryPoints * kNeighbors]),
                                                                                                                                       dists_ptr(new T[nqueryPoints * kNeighbors]),
                                                                                                                                       indices(indices_ptr.get(), nqueryPoints, kNeighbors),
                                                                                                                                       dists(dists_ptr.get(), nqueryPoints, kNeighbors),
                                                                                                                                       the_same(false)
{
    if (drows < static_cast<std::size_t>(nn))
    {
        UMUQFAIL("Not enough points to create K nearest neighbors for each point !");
    }
}

template <typename T, class Distance>
kNearestNeighbor<T, Distance>::kNearestNeighbor(kNearestNeighbor<T, Distance> &&other) : drows(other.drows),
                                                                                         qrows(other.qrows),
                                                                                         cols(other.cols),
                                                                                         nn(other.nn),
                                                                                         indices_ptr(std::move(other.indices_ptr)),
                                                                                         dists_ptr(std::move(other.dists_ptr)),
                                                                                         indices(std::move(other.indices)),
                                                                                         dists(std::move(other.dists)),
                                                                                         the_same(other.the_same)
{
}

template <typename T, class Distance>
kNearestNeighbor<T, Distance>::kNearestNeighbor(kNearestNeighbor<T, Distance> const &other) : drows(other.drows),
                                                                                              qrows(other.qrows),
                                                                                              cols(other.cols),
                                                                                              nn(other.nn),
                                                                                              indices_ptr(new int[other.qrows * other.nn]),
                                                                                              dists_ptr(new T[other.qrows * other.nn]),
                                                                                              indices(indices_ptr.get(), other.qrows, other.nn),
                                                                                              dists(dists_ptr.get(), other.qrows, other.nn),
                                                                                              the_same(other.the_same)
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

template <typename T, class Distance>
kNearestNeighbor<T, Distance> &kNearestNeighbor<T, Distance>::operator=(kNearestNeighbor<T, Distance> &&other)
{
    drows = std::move(other.drows);
    qrows = std::move(other.qrows);
    cols = std::move(other.cols);
    nn = std::move(other.nn);
    the_same = std::move(other.the_same);
    indices_ptr = std::move(other.indices_ptr);
    dists_ptr = std::move(other.dists_ptr);
    indices = std::move(other.indices);
    dists = std::move(other.dists);
    return *this;
}

template <typename T, class Distance>
kNearestNeighbor<T, Distance>::~kNearestNeighbor() {}

template <typename T, class Distance>
void kNearestNeighbor<T, Distance>::buildIndex(T *idata)
{
    flann::Matrix<T> dataset(idata, drows, cols);

    // Construct an randomized kd-tree index using 4 kd-trees
    // For the number of parallel kd-trees to use (Good values are in the range [1..16])
    flann::Index<Distance> index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();

    // Do a knn search, using 128 checks
    // Number of checks means: How many leafs to visit when searching
    // for neighbors (-1 for unlimited)
    index.knnSearch(dataset, indices, dists, nn, flann::SearchParams(128));
}

template <typename T, class Distance>
void kNearestNeighbor<T, Distance>::buildIndex(T *idata, T *qdata)
{
    flann::Matrix<T> dataset(idata, drows, cols);

    // Construct an randomized kd-tree index using 4 kd-trees
    // For the number of parallel kd-trees to use (Good values are in the range [1..16])
    flann::Index<Distance> index(dataset, flann::KDTreeIndexParams(4));
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

template <typename T, class Distance>
inline int *kNearestNeighbor<T, Distance>::NearestNeighbors(int const &index) const
{
    // +1 is that we do not want the index of the point itself
    return indices_ptr.get() + index * nn + the_same;
}

template <typename T, class Distance>
inline int *kNearestNeighbor<T, Distance>::NearestNeighbors() const
{
    return indices_ptr.get();
}

template <typename T, class Distance>
inline T *kNearestNeighbor<T, Distance>::NearestNeighborsDistances(int const &index) const
{
    // +1 is that we do not want the index of the point itself
    return dists_ptr.get() + index * nn + the_same;
}

template <typename T, class Distance>
inline T kNearestNeighbor<T, Distance>::minDist(int const &index) const
{
    std::ptrdiff_t const Id = index * nn + the_same;
    return dists_ptr[Id];
}

template <typename T, class Distance>
inline T *kNearestNeighbor<T, Distance>::minDist()
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

template <typename T, class Distance>
inline int kNearestNeighbor<T, Distance>::numNearestNeighbors() const
{
    return nn - the_same;
}

template <typename T, class Distance>
bool kNearestNeighbor<T, Distance>::checkNearestNeighbors()
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

template <typename T, class Distance>
inline void kNearestNeighbor<T, Distance>::IndexSwap(int Indx1, int Indx2)
{
    std::swap(indices_ptr[Indx1], indices_ptr[Indx2]);
    std::swap(dists_ptr[Indx1], dists_ptr[Indx2]);
}

template <typename T, class Distance>
inline int kNearestNeighbor<T, Distance>::numInputdata() const
{
    return drows;
}

template <typename T, class Distance>
inline int kNearestNeighbor<T, Distance>::numQuerydata() const
{
    return qrows;
}

/*!
 * \ingroup Numerics_Module
 * \todo
 * Somehow the specialized template did not work.
 * FIXME: to the correct templated version
 */

/*! \class L2NearestNeighbor
 * \ingroup Numerics_Module
 * 
 * \brief Finding K nearest neighbors in high dimensional spaces using L2 distance functor
 * 
 * \tparam T Data type
 */
template <typename T>
class L2NearestNeighbor : public kNearestNeighbor<T, flann::L2<T>>
{
  public:
    /*!
     * \brief Construct a new L2NearestNeighbor object
     * 
     * \param ndataPoints  Number of data points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    L2NearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Construct a new L2NearestNeighbor object
     * 
     * \param ndataPoints  Number of data points
     * \param nqueryPoints Number of query points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    L2NearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Move construct a new L2NearestNeighbor object
     * 
     * \param other L2NearestNeighbor object
     */
    L2NearestNeighbor(L2NearestNeighbor<T> &&other);

    /*!
     * \brief Copy construct a new L2NearestNeighbor object
     * 
     * \param other L2NearestNeighbor object
     */
    L2NearestNeighbor(L2NearestNeighbor<T> const &other);

    /*!
     * \brief Move assignment
     * 
     * \param other L2NearestNeighbor object
     * \returns L2NearestNeighbor<T>& 
     */
    L2NearestNeighbor<T> &operator=(L2NearestNeighbor<T> &&other);
};

template <typename T>
L2NearestNeighbor<T>::L2NearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
L2NearestNeighbor<T>::L2NearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
L2NearestNeighbor<T>::L2NearestNeighbor(L2NearestNeighbor<T> &&other) : kNearestNeighbor<T, flann::L2<T>>(std::move(other)) {}

template <typename T>
L2NearestNeighbor<T>::L2NearestNeighbor(L2NearestNeighbor<T> const &other) : kNearestNeighbor<T, flann::L2<T>>(other) {}

template <typename T>
L2NearestNeighbor<T> &L2NearestNeighbor<T>::operator=(L2NearestNeighbor<T> &&other)
{
    this->drows = std::move(other.drows);
    this->qrows = std::move(other.qrows);
    this->cols = std::move(other.cols);
    this->nn = std::move(other.nn);
    this->the_same = std::move(other.the_same);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);

    return static_cast<L2NearestNeighbor<T> &>(kNearestNeighbor<T, flann::L2<T>>::operator=(std::move(other)));
}

/*! \class MahalanobisNearestNeighbor
 * \ingroup Numerics_Module
 * 
 * \brief Finding K nearest neighbors in high dimensional spaces using Mahalanobis distance.
 * 
 * This class finds the K nearest neighbors in high dimensional spaces using Mahalanobis distance.
 * The Mahalanobis distance is a measure of the distance between a point and a distribution.
 * It is a multi-dimensional generalization of the idea of measuring how many standard deviations 
 * away the point is from the mean of the distribution. <br>
 * It is essentially a stretched Euclidean distance.
 * 
 * The Mahalanobis distance between two vectors \f$ \vec{x} \f$ and \f$ \vec{y} \f$ with dimensionality 
 * of \f$ d, \f$ from the same distribution with the covariance matrix \f$ \Sigma \f$ of size 
 * \f$ d \times d \f$ is:<br>
 * 
 * \f$ d(\vec{x}, \vec{y}) = \sqrt{(\vec{x} - \vec{y})^T \Sigma^{-1} (\vec{x} - \vec{y})}.\f$ 
 *
 * In this implamentation, we use the L2 distance and simply stretch the actual dataset
 * itself before performing any evaluations. This is done to avoid calculating the full 
 * rating matrix, for Mahalanobis distance metric in K nearest neighbor search. 
 * 
 * We do nearest neighbor search only on the transformed space, where we stretch the actual 
 * dataset itself before performing any evaluations.
 * 
 * \f$
 * \begin{align}
 * \nonumber \text{Square of the Mahalanobis distance} &=  \\
 * \nonumber d(\vec{x}, \vec{y})^2                     &= (\vec{x} - \vec{y})^T \Sigma^{-1} \underbrace{(\vec{x} - \vec{y})}_{\vec{X}} \\
 * \nonumber d(\vec{x}, \vec{y})^2                     &= \vec{X}^T\Sigma^{-1}\vec{X} \\
 * \nonumber \text{by decomposing}~\Sigma^{-1}=LL^T \text{(Cholesky decomposition)} \\
 * \nonumber                                           &= \vec{X}^T LL^T\vec{X} \\
 * \nonumber                                           &={(L^T\vec{X})}^T{(L^T\vec{X})}
 * \end{align}
 * \f$
 *
 * This can be seen as the nearest neighbor search on the \f$ L^T\vec{X} \f$ with
 * using the squared Euclidean distance. <br> 
 * 
 * 
 * \warning
 * Avoid doing the Cholesky decomposition at first, and inversion later. Since, it would result to numerical inaccuracy.<br>
 * Doing this some of the neighbors are computed wrongly.<br>
 * \f$
 * \begin{align}
 * \nonumber d(\vec{x}, \vec{y})^2                     &= \vec{X}^T\Sigma^{-1}\vec{X} \\
 * \nonumber \text{by decomposing}~\Sigma=LL^T \text{(Cholesky decomposition)} \\
 * \nonumber                                           &= \vec{X}^T (LL^T)^{-1}\vec{X} \\
 * \nonumber                                           &= \vec{X}^T L^{-T}L^{-1}\vec{X} \\
 * \nonumber                                           &={(L^{-1}\vec{X})}^T{(L^{-1}\vec{X})}
 * \end{align}
 * \f$
 * 
 * 
 * \todo
 * The Mahalanobis distance should be integrated in FLANN framework for efficiency.
 *
 * \tparam T Data type
 */
template <typename T>
class MahalanobisNearestNeighbor : public kNearestNeighbor<T, flann::L2<T>>
{
  public:
    /*!
     * \brief Construct a new MahalanobisNearestNeighbor object
     * 
     * \param ndataPoints  Number of data points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    MahalanobisNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Construct a new MahalanobisNearestNeighbor object
     * 
     * \param ndataPoints  Number of data points
     * \param nqueryPoints Number of query points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    MahalanobisNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Move construct a new MahalanobisNearestNeighbor object
     * 
     * \param other MahalanobisNearestNeighbor object
     */
    MahalanobisNearestNeighbor(MahalanobisNearestNeighbor<T> &&other);

    /*!
     * \brief Copy construct a new MahalanobisNearestNeighbor object
     * 
     * \param other MahalanobisNearestNeighbor object
     */
    MahalanobisNearestNeighbor(MahalanobisNearestNeighbor<T> const &other);

    /*!
     * \brief Move assignment 
     * 
     * \param other MahalanobisNearestNeighbor object
     * \returns MahalanobisNearestNeighbor<T>& 
     */
    MahalanobisNearestNeighbor<T> &operator=(MahalanobisNearestNeighbor<T> &&other);

    /*!
     * \brief Construct a kd-tree index & do a knn search
     * 
     * \param idata A pointer to input data 
     */
    void buildIndex(T *idata);

    /*!
     * \brief Construct a kd-tree index & do a knn search
     * 
     * \param idata A pointer to input data 
     * \param qdata A pointer to query data 
     */
    void buildIndex(T *idata, T *qdata);

    /*!
     * \brief Set the Covariance object
     * 
     * \param Covariance  The covariance matrix
     */
    void setCovariance(EMatrixX<T> const &Covariance);

    /*!
     * \brief Set the Covariance object
     * 
     * \param Covariance  The covariance matrix
     */
    void setCovariance(T const *Covariance);

    /*!
     * \brief Set the Covariance object
     * 
     * \param Covariance  The covariance matrix
     */
    void setCovariance(std::vector<T> const &Covariance);

    /*!
     * \brief Access the covariance matrix.
     *
     * \returns Constant reference to the covariance matrix.
     */
    EMatrixX<T> const &Covariance() const;

    /*!
     * \brief Modify the covariance matrix.
     *
     * \returns Reference to the covariance matrix.
     */
    EMatrixX<T> &Covariance();

  protected:
    /*!
     * \brief The covariance matrix \f$ \Sigma \f$ associated with Mahalanobis distance.
     * 
     * To avoid calculating the full rating matrix, for Mahalanobis distance metric 
     * in K nearest neighbor search, we will do nearest neighbor search only on the 
     * transformed space, where we stretch the actual dataset itself before performing 
     * any evaluations.
     * 
     * \f$
     * \begin{align}
     * \nonumber \text{Square of the Mahalanobis distance} &=  \\
     * \nonumber d(\vec{x}, \vec{y})^2                     &= (\vec{x} - \vec{y})^T \Sigma^{-1} \underbrace{(\vec{x} - \vec{y})}_{\vec{X}} \\
     * \nonumber d(\vec{x}, \vec{y})^2                     &= \vec{X}^T\Sigma^{-1}\vec{X} \\
     * \nonumber \text{by decomposing}~\Sigma^{-1}=LL^T \text{(Cholesky decomposition)} \\
     * \nonumber                                           &= \vec{X}^T LL^T\vec{X} \\
     * \nonumber                                           &={(L^T\vec{X})}^T{(L^T\vec{X})}
     * \end{align}
     * \f$
     *
     * This can be seen as the nearest neighbor search on the \f$ L^T\vec{X} \f$ with 
     * using the squared Euclidean distance. <br> 
     * So, we'll decompose \f$ \Sigma^{-1} = LL^T \f$ (the Cholesky decomposition), 
     * and then multiply sampling space points and query points by \f$ L^T \f$. 
     * Then we can perform nearest neighbor search using the squared Euclidean distance.
     *
     */
    EMatrixX<T> covariance;

    //! A view of the lower triangular matrix L from the Cholesky decomposition of inverse covariance matrix
    EMatrixX<T> matrixL;
};

template <typename T>
MahalanobisNearestNeighbor<T>::MahalanobisNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nDim, kNeighbors),
                                                                                                                         covariance(EMatrixX<T>::Identity(nDim, nDim)),
                                                                                                                         matrixL(EMatrixX<T>::Identity(nDim, nDim))
{
    if (!(unrolledIncrement == 0 || unrolledIncrement == 4 || unrolledIncrement == 6 || unrolledIncrement == 8 || unrolledIncrement == 10 || unrolledIncrement == 12))
    {
        UMUQFAIL("The unrolled increment value of unrolledIncrement is not correctly set!");
    }
}

template <typename T>
MahalanobisNearestNeighbor<T>::MahalanobisNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors),
                                                                                                                                                 covariance(EMatrixX<T>::Identity(nDim, nDim)),
                                                                                                                                                 matrixL(EMatrixX<T>::Identity(nDim, nDim))
{
    if (!(unrolledIncrement == 0 || unrolledIncrement == 4 || unrolledIncrement == 6 || unrolledIncrement == 8 || unrolledIncrement == 10 || unrolledIncrement == 12))
    {
        UMUQFAIL("The unrolled increment value of unrolledIncrement is not correctly set!");
    }
}

template <typename T>
MahalanobisNearestNeighbor<T>::MahalanobisNearestNeighbor(MahalanobisNearestNeighbor<T> &&other) : kNearestNeighbor<T, flann::L2<T>>(std::move(other))
{
    this->covariance = std::move(other.covariance);
    this->matrixL = std::move(other.matrixL);
}

template <typename T>
MahalanobisNearestNeighbor<T>::MahalanobisNearestNeighbor(MahalanobisNearestNeighbor<T> const &other) : kNearestNeighbor<T, flann::L2<T>>(other)
{
    this->covariance = other.covariance;
    this->matrixL = other.matrixL;
}

template <typename T>
MahalanobisNearestNeighbor<T> &MahalanobisNearestNeighbor<T>::operator=(MahalanobisNearestNeighbor<T> &&other)
{
    this->drows = std::move(other.drows);
    this->qrows = std::move(other.qrows);
    this->cols = std::move(other.cols);
    this->nn = std::move(other.nn);
    this->the_same = std::move(other.the_same);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->covariance = std::move(other.covariance);
    this->matrixL = std::move(other.matrixL);

    return static_cast<MahalanobisNearestNeighbor<T> &>(kNearestNeighbor<T, flann::L2<T>>::operator=(std::move(other)));
}

template <typename T>
void MahalanobisNearestNeighbor<T>::buildIndex(T *idata)
{
    if (covariance.isIdentity(1e-3))
    {
        kNearestNeighbor<T, flann::L2<T>>::buildIndex(idata);
    }
    else
    {
        // Copy the input data into a temporary array
        std::vector<T> inputData{idata, idata + this->drows * this->cols};

        // Map the temporary array in Eigen format without memory copy
        EMapType<T> inputDataEMap(inputData.data(), this->drows, this->cols);

        // Compute the input data matrix multiply by lower triangular matrix L from
        // the Cholesky decomposition of inverse covariance matrix
        inputDataEMap *= matrixL;

        // Map the data in flann matrix format
        flann::Matrix<T> dataset(inputData.data(), this->drows, this->cols);

        // Construct an randomized kd-tree index using 4 kd-trees
        // For the number of parallel kd-trees to use (Good values are in the range [1..16])
        flann::Index<flann::L2<T>> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        // Do a knn search, using 128 checks
        // Number of checks means: How many leafs to visit when searching
        // for neighbors (-1 for unlimited)
        index.knnSearch(dataset, this->indices, this->dists, this->nn, flann::SearchParams(128));

        // Total number of nearest neighbors for each point
        int nNN = this->numNearestNeighbors();

        // Correct the distances
        // Loop over all points
        for (int i = 0; i < this->drows; i++)
        {
            std::ptrdiff_t const IdI = i * this->cols;

            // A pointer to nearest neighbors indices of point i
            int *NearestNeighbors = this->NearestNeighbors(i);

            // A pointer to nearest neighbors square distances from the point i
            T *nnDist = this->NearestNeighborsDistances(i);

#if unrolledIncrement == 0
            {
                T *last = idata + IdI + this->cols;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->cols;

                    // pointer to query data
                    T *Idata = idata + IdI;

                    // pointer to idata (neighbors of i)
                    T *Jdata = idata + IdJ;

                    T result(0);
                    while (Idata < last)
                    {
                        T const diff0 = *Idata++ - *Jdata++;
                        result += diff0 * diff0;
                    }

                    nnDist[j] = result;
                }
            }
#else
            {
                T *last = idata + IdI + this->cols;
                T *lastgroup = last - unrolledIncrement + 1;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->cols;

                    // pointer to query data
                    T *Idata = idata + IdI;

                    // pointer to idata (neighbors of i)
                    T *Jdata = idata + IdJ;

                    T result(0);

                    // Process unrolled Increment items with each loop for efficiency.
                    while (Idata < lastgroup)
                    {
                        T const diff0 = Idata[0] - Jdata[0];
                        T const diff1 = Idata[1] - Jdata[1];
                        T const diff2 = Idata[2] - Jdata[2];
                        T const diff3 = Idata[3] - Jdata[3];
#if unrolledIncrement == 4
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
#endif
#if unrolledIncrement == 6
                        T const diff4 = Idata[4] - Jdata[4];
                        T const diff5 = Idata[5] - Jdata[5];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5;
#endif
#if unrolledIncrement == 8
                        T const diff4 = Idata[4] - Jdata[4];
                        T const diff5 = Idata[5] - Jdata[5];
                        T const diff6 = Idata[6] - Jdata[6];
                        T const diff7 = Idata[7] - Jdata[7];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
#endif
#if unrolledIncrement == 10
                        T const diff4 = Idata[4] - Jdata[4];
                        T const diff5 = Idata[5] - Jdata[5];
                        T const diff6 = Idata[6] - Jdata[6];
                        T const diff7 = Idata[7] - Jdata[7];
                        T const diff8 = Idata[8] - Jdata[8];
                        T const diff9 = Idata[9] - Jdata[9];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9;
#endif
#if unrolledIncrement == 12
                        T const diff4 = Idata[4] - Jdata[4];
                        T const diff5 = Idata[5] - Jdata[5];
                        T const diff6 = Idata[6] - Jdata[6];
                        T const diff7 = Idata[7] - Jdata[7];
                        T const diff8 = Idata[8] - Jdata[8];
                        T const diff9 = Idata[9] - Jdata[9];
                        T const diff10 = Idata[10] - Jdata[10];
                        T const diff11 = Idata[11] - Jdata[11];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9 + diff10 * diff10 + diff11 * diff11;
#endif
                        Idata += unrolledIncrement;
                        Jdata += unrolledIncrement;
                    }
                    // Process last pixels.
                    while (Idata < last)
                    {
                        T const diff0 = *Idata++ - *Jdata++;
                        result += diff0 * diff0;
                    }

                    nnDist[j] = result;
                }
            }
#endif
        }
    }
}

template <typename T>
void MahalanobisNearestNeighbor<T>::buildIndex(T *idata, T *qdata)
{
    if (covariance.isIdentity(1e-3))
    {
        kNearestNeighbor<T, flann::L2<T>>::buildIndex(idata, qdata);
    }
    else
    {
        // Copy the input data into a temporary array
        std::vector<T> inputData{idata, idata + this->drows * this->cols};

        // Copy the query data into a temporary array
        std::vector<T> queryData{qdata, qdata + this->qrows * this->cols};

        // Map the temporary array in Eigen format without memory copy
        EMapType<T> inputDataEMap(inputData.data(), this->drows, this->cols);

        // Map the temporary array in Eigen format without memory copy
        EMapType<T> queryDataEMap(queryData.data(), this->qrows, this->cols);

        // Compute the input data matrix multiply by lower triangular matrix L from
        // the Cholesky decomposition of inverse covariance matrix
        inputDataEMap *= matrixL;

        queryDataEMap *= matrixL;

        // Map the data in flann matrix format
        flann::Matrix<T> dataset(inputData.data(), this->drows, this->cols);

        // Construct an randomized kd-tree index using 4 kd-trees
        // For the number of parallel kd-trees to use (Good values are in the range [1..16])
        flann::Index<flann::L2<T>> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        flann::Matrix<T> query(queryData.data(), this->qrows, this->cols);

        // Do a knn search, using 128 checks
        // Number of checks means: How many leafs to visit when searching
        // for neighbors (-1 for unlimited)
        index.knnSearch(query, this->indices, this->dists, this->nn, flann::SearchParams(128));

        if (!this->checkNearestNeighbors())
        {
            UMUQWARNING("Input data & query data are the same!");
        }

        // Total number of nearest neighbors for each point
        int const nNN = this->numNearestNeighbors();

        // Correct the distances
        // Loop over all query points
        for (int i = 0; i < this->qrows; i++)
        {
            std::ptrdiff_t const IdI = i * this->cols;

            // A pointer to nearest neighbors indices of point i
            int *NearestNeighbors = this->NearestNeighbors(i);

            // A pointer to nearest neighbors square distances from the point i
            T *nnDist = this->NearestNeighborsDistances(i);

#if unrolledIncrement == 0
            {
                T *last = qdata + IdI + this->cols;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->cols;

                    // pointer to query data
                    T *Idata = qdata + IdI;

                    // pointer to idata (neighbors of i)
                    T *Jdata = idata + IdJ;

                    T result(0);
                    while (Idata < last)
                    {
                        T const diff0 = *Idata++ - *Jdata++;
                        result += diff0 * diff0;
                    }

                    nnDist[j] = result;
                }
            }
#else
            {
                T *last = qdata + IdI + this->cols;
                T *lastgroup = last - unrolledIncrement + 1;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->cols;

                    // pointer to query data
                    T *Idata = qdata + IdI;

                    // pointer to idata (neighbors of i)
                    T *Jdata = idata + IdJ;

                    T result(0);

                    // Process 4 items with each loop for efficiency.
                    while (Idata < lastgroup)
                    {
                        T const diff0 = Idata[0] - Jdata[0];
                        T const diff1 = Idata[1] - Jdata[1];
                        T const diff2 = Idata[2] - Jdata[2];
                        T const diff3 = Idata[3] - Jdata[3];
#if unrolledIncrement == 4
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
#endif
#if unrolledIncrement == 6
                        T const diff4 = Idata[4] - Jdata[4];
                        T const diff5 = Idata[5] - Jdata[5];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5;
#endif
#if unrolledIncrement == 8
                        T const diff4 = Idata[4] - Jdata[4];
                        T const diff5 = Idata[5] - Jdata[5];
                        T const diff6 = Idata[6] - Jdata[6];
                        T const diff7 = Idata[7] - Jdata[7];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
#endif
#if unrolledIncrement == 10
                        T const diff4 = Idata[4] - Jdata[4];
                        T const diff5 = Idata[5] - Jdata[5];
                        T const diff6 = Idata[6] - Jdata[6];
                        T const diff7 = Idata[7] - Jdata[7];
                        T const diff8 = Idata[8] - Jdata[8];
                        T const diff9 = Idata[9] - Jdata[9];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9;
#endif
#if unrolledIncrement == 12
                        T const diff4 = Idata[4] - Jdata[4];
                        T const diff5 = Idata[5] - Jdata[5];
                        T const diff6 = Idata[6] - Jdata[6];
                        T const diff7 = Idata[7] - Jdata[7];
                        T const diff8 = Idata[8] - Jdata[8];
                        T const diff9 = Idata[9] - Jdata[9];
                        T const diff10 = Idata[10] - Jdata[10];
                        T const diff11 = Idata[11] - Jdata[11];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9 + diff10 * diff10 + diff11 * diff11;
#endif
                        Idata += unrolledIncrement;
                        Jdata += unrolledIncrement;
                    }
                    // Process last 0-3 pixels.
                    while (Idata < last)
                    {
                        T const diff0 = *Idata++ - *Jdata++;
                        result += diff0 * diff0;
                    }

                    nnDist[j] = result;
                }
            }
#endif
        }
    }
}

template <typename T>
void MahalanobisNearestNeighbor<T>::setCovariance(EMatrixX<T> const &Covariance)
{
    covariance = Covariance;

    // check to see if the input covariance matrix is positive definite, or not
    if (!isSelfAdjointMatrixPositiveDefinite<EMatrixX<T>>(covariance))
    {
        // If the input covariance matrix is not positive definite, we force it
        forceSelfAdjointMatrixPositiveDefinite<EMatrixX<T>>(covariance);
    }

    // Inverse the positive definite covariance matrix
    EMatrixX<T> covarianceInverse = covariance.inverse();

    // Compute the Cholesky decomposition \f$ (LL^T) \f$ and retrieve factor \f$ L \f$ in the decomposition.
    matrixL = covarianceInverse.llt().matrixL();
}

template <typename T>
void MahalanobisNearestNeighbor<T>::setCovariance(std::vector<T> const &Covariance)
{
    covariance = EMapType<T>(Covariance.data(), this->cols, this->cols);

    // check to see if the input covariance matrix is positive definite, or not
    if (!isSelfAdjointMatrixPositiveDefinite<EMatrixX<T>>(covariance))
    {
        // If the input covariance matrix is not positive definite, we force it
        forceSelfAdjointMatrixPositiveDefinite<EMatrixX<T>>(covariance);
    }

    // Inverse the positive definite covariance matrix
    EMatrixX<T> covarianceInverse = covariance.inverse();

    // Compute the Cholesky decomposition \f$ (LL^T) \f$ and retrieve factor \f$ L \f$ in the decomposition.
    matrixL = covarianceInverse.llt().matrixL();
}

template <typename T>
void MahalanobisNearestNeighbor<T>::setCovariance(T const *Covariance)
{
    covariance = EMapType<T>(Covariance, this->cols, this->cols);

    // check to see if the input covariance matrix is positive definite, or not
    if (!isSelfAdjointMatrixPositiveDefinite<EMatrixX<T>>(covariance))
    {
        // If the input covariance matrix is not positive definite, we force it
        forceSelfAdjointMatrixPositiveDefinite<EMatrixX<T>>(covariance);
    }

    // Inverse the positive definite covariance matrix
    EMatrixX<T> covarianceInverse = covariance.inverse();

    // Compute the Cholesky decomposition \f$ (LL^T) \f$ and retrieve factor \f$ L \f$ in the decomposition.
    matrixL = covarianceInverse.llt().matrixL();
}

template <typename T>
EMatrixX<T> const &MahalanobisNearestNeighbor<T>::Covariance() const { return this->covariance; }

template <typename T>
EMatrixX<T> &MahalanobisNearestNeighbor<T>::Covariance() { return this->covariance; }

} // namespace umuq

#endif // UMUQ_KNEARESTNEIGHBORS
