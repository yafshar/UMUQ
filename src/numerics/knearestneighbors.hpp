#ifndef UMUQ_KNEARESTNEIGHBORS_H
#define UMUQ_KNEARESTNEIGHBORS_H

/*!
 * FLANN is a library for performing fast approximate nearest neighbor searches in high dimensional spaces. 
 * It contains a collection of algorithms we found to work best for nearest neighbor search and a system 
 * for automatically choosing the best algorithm and optimum parameters depending on the dataset.
 */
#include <flann/flann.hpp>

namespace umuq
{

/*! \class kNearestNeighbor
 * \brief Finding K nearest neighbors in high dimensional spaces
 * 
 * \tparam T         data type
 * \tparam Distance  Distance type for computing the distances to the nearest neighbors
 *                   (Default is a specialized class \b kNearestNeighbor<T> with L2 distance)
 * 
 * \b EUCLIDEAN      Squared Euclidean distance functor, optimized version 
 * \b L2             Squared Euclidean distance functor, optimized version 
 * \b MANHATTAN      Manhattan distance functor, optimized version
 * \b L1             Manhattan distance functor, optimized version
 * \b L2_SIMPLE      Squared Euclidean distance functor
 * \b MINKOWSKI      The Minkowsky (L_p) distance between two vectors.
 * \b MAX
 * \b HIST_INTERSECT
 * \b HELLINGER      The Hellinger distance, quantify the similarity between two probability distributions.
 * \b CHI_SQUARE     The distance between two histograms
 * \b KULLBACK_LEIBLER
 * \b HAMMING
 * \b HAMMING_LUT    Hamming distance functor - counts the bit differences between two strings - 
 *                   useful for the Brief descriptor bit count of A exclusive XOR'ed with B
 * \b HAMMING_POPCNT Hamming distance functor (pop count between two binary vectors, i.e. xor them 
 *                   and count the number of bits set)
 */
template <typename T, class Distance>
class kNearestNeighbor
{
  public:
    /*!
     * \brief constructor
     * 
     * \param ndataPoints Number of data points
     * \param nDim        Dimension of each point
     * \param nN          Number of nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nDim, int const nN);

    /*!
     * \brief constructor
     * 
     * \param ndataPoints  Number of data points
     * \param nqueryPoints Number of query points
     * \param nDim         Dimension of each point
     * \param nN           Number of nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const nN);

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
    void buildIndex(T *idata);

    /*!
     * \brief Construct a kd-tree index & do a knn search
     * 
     * \param idata A pointer to input data 
     * \param qdata A pointer to query data 
     */
    void buildIndex(T *idata, T *qdata);

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
     * The function returns a pointer of size(nPoints * (nN+1)) 
     * all points neighbors.
     * The retorned pointer looks like below:
     *    0                1      .     nN
     *   ---------------------------------
     *  | 0               0_1     .     0_nN
     *  | 1               1_1     .     1_nN
     *  | .
     *  | nPoints-1        .      .     (nPoints-1)_nN
     * 
     * Each row has the size of nn which is the number of neighbors + 1
     * and it has nPoints rows.
     * The first column is the indices of points themselves.
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
     * \brief  Nmber of each point nearest neighbors
     * 
     * \returns number of nearest neighbors
     */
    inline int numNearestNeighbors() const;

    /*!
     * \brief   Function to make sure that we do not compute the nearest neighbors of a point from itself
     * 
     * \returns true for if input points and query points are used correctly
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

  private:
    //! Number of data rows
    std::size_t drows;

    //! Number of query rows
    std::size_t qrows;

    //! Number of columns
    std::size_t cols;

    //! Number of nearest neighbors to find
    int nn;

    std::unique_ptr<int[]> indices_ptr;
    std::unique_ptr<T[]> dists_ptr;

    flann::Matrix<int> indices;
    flann::Matrix<T> dists;

    //! Flag to check if the input data and qury data are the same
    bool the_same;
};

template <typename T, class Distance>
kNearestNeighbor<T, Distance>::kNearestNeighbor(int const ndataPoints, int const nDim, int const nN) : drows(ndataPoints),
                                                                                                       qrows(ndataPoints),
                                                                                                       cols(nDim),
                                                                                                       nn(nN + 1),
                                                                                                       indices_ptr(new int[ndataPoints * (nN + 1)]),
                                                                                                       dists_ptr(new T[ndataPoints * (nN + 1)]),
                                                                                                       indices(indices_ptr.get(), ndataPoints, (nN + 1)),
                                                                                                       dists(dists_ptr.get(), ndataPoints, (nN + 1)),
                                                                                                       the_same(true)
{
    if (drows < static_cast<std::size_t>(nn))
    {
        UMUQFAIL("Not enough points to create K nearest neighbors for each point !");
    }
}

template <typename T, class Distance>
kNearestNeighbor<T, Distance>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const nN) : drows(ndataPoints),
                                                                                                                               qrows(nqueryPoints),
                                                                                                                               cols(nDim),
                                                                                                                               nn(nN),
                                                                                                                               indices_ptr(new int[nqueryPoints * nN]),
                                                                                                                               dists_ptr(new T[nqueryPoints * nN]),
                                                                                                                               indices(indices_ptr.get(), nqueryPoints, nN),
                                                                                                                               dists(dists_ptr.get(), nqueryPoints, nN),
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
    // for neighbours (-1 for unlimited)
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
    // for neighbours (-1 for unlimited)
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
    if (s == qrows)
    {
        return false;
    }
    return true;
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

// TODO : Somehow the specialized template did not work.
// FIXME: to the correct templated version

/*! \class L2NearestNeighbor
 * \brief Finding K nearest neighbors in high dimensional spaces using L2 distance functor
 * 
 * \tparam T data type
 */
template <typename T>
class L2NearestNeighbor : public kNearestNeighbor<T, flann::L2<T>>
{
  public:
    L2NearestNeighbor(int const ndataPoints, int const nDim, int const nN);
    L2NearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const nN);
    L2NearestNeighbor(L2NearestNeighbor<T> &&other);
    L2NearestNeighbor(L2NearestNeighbor<T> const &other);
    L2NearestNeighbor<T> &operator=(L2NearestNeighbor<T> &&other);
};

template <typename T>
L2NearestNeighbor<T>::L2NearestNeighbor(int const ndataPoints, int const nDim, int const nN) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nDim, nN)
{
}
template <typename T>
L2NearestNeighbor<T>::L2NearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const nN) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nqueryPoints, nDim, nN) {}

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

// namespace flann
// {
// /*!
//  * \brief covariance variable to be used inside flann
//  *
//  * \tparam T Data type
//  */
// template <class T>
// std::unique_ptr<T[]> covariance;

// std::size_t covarianceIdx;

// /*! \class Mahalanobis
//  * \brief Mahalanobis distance functor
//  *
//  */
// template <class T>
// struct Mahalanobis
// {
//     typedef bool is_kdtree_distance;
//     typedef T ElementType;
//     typedef typename Accumulator<T>::Type ResultType;

//     /*!
//      * \brief Compute the Mahalanobis distance between two vectors.
//      *
//      *
//      */
//     template <typename Iterator1, typename Iterator2>
//     ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
//     {
//         ResultType result = ResultType();
//         ResultType diff0, diff1, diff2, diff3;
//         Iterator1 last = a + size;
//         Iterator1 lastgroup = last - 3;

//         arrayWrapper<T> cArray(covariance.get() + covarianceIdx, size * size, size);

//         /* Process 4 items with each loop for efficiency. */
//         while (a < lastgroup)
//         {
//             diff0 = (ResultType)(a[0] - b[0]);
//             diff1 = (ResultType)(a[1] - b[1]);
//             diff2 = (ResultType)(a[2] - b[2]);
//             diff3 = (ResultType)(a[3] - b[3]);
//             result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
//             a += 4;
//             b += 4;

//             if ((worst_dist > 0) && (result > worst_dist))
//             {
//                 return result;
//             }
//         }
//         /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
//         while (a < last)
//         {
//             diff0 = (ResultType)(*a++ - *b++);
//             result += diff0 * diff0;
//         }
//         return result;
//     }

//     /**
//      *	Partial euclidean distance, using just one dimension. This is used by the
//      *	kd-tree when computing partial distances while traversing the tree.
//      *
//      *	Squared root is omitted for efficiency.
//      */
//     template <typename U, typename V>
//     inline ResultType accum_dist(const U &a, const V &b, int) const
//     {
//         return (a - b) * (a - b);
//     }
// };
// }

} // namespace umuq

#endif // UMUQ_FLANNLIB_H
