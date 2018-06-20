#ifndef UMUQ_KNEARESTNEIGHBORS_H
#define UMUQ_KNEARESTNEIGHBORS_H

/*!
 * FLANN is a library for performing fast approximate nearest neighbor searches in high dimensional spaces. 
 * It contains a collection of algorithms we found to work best for nearest neighbor search and a system 
 * for automatically choosing the best algorithm and optimum parameters depending on the dataset.
 */
#include <flann/flann.hpp>

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
    kNearestNeighbor(int const ndataPoints, int const nDim, int const nN) : drows(ndataPoints),
                                                                            qrows(ndataPoints),
                                                                            cols(nDim),
                                                                            nn(nN + 1),
                                                                            indices_ptr(new int[ndataPoints * (nN + 1)]),
                                                                            dists_ptr(new T[ndataPoints * (nN + 1)]),
                                                                            indices(indices_ptr.get(), ndataPoints, (nN + 1)),
                                                                            dists(dists_ptr.get(), ndataPoints, (nN + 1)),
                                                                            the_same(true)
    {
        if (drows < nn)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Not enough points to create " << nN << " nearest neighbors for each point !" << std::endl;
            throw(std::runtime_error("Not enough points!"));
        }
    }

    /*!
     * \brief constructor
     * 
     * \param ndataPoints  Number of data points
     * \param nqueryPoints Number of query points
     * \param nDim         Dimension of each point
     * \param nN           Number of nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const nN) : drows(ndataPoints),
                                                                                                    qrows(nqueryPoints),
                                                                                                    cols(nDim),
                                                                                                    nn(nN),
                                                                                                    indices_ptr(new int[nqueryPoints * nN]),
                                                                                                    dists_ptr(new T[nqueryPoints * nN]),
                                                                                                    indices(indices_ptr.get(), nqueryPoints, nN),
                                                                                                    dists(dists_ptr.get(), nqueryPoints, nN),
                                                                                                    the_same(false)
    {
        if (drows < nn)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Not enough points to create " << nN << " nearest neighbors for each point !" << std::endl;
            throw(std::runtime_error("Not enough points!"));
        }
    }

    /*!
     * \brief Move constructor
     * \param inputObj kNearestNeighbor to be moved
     */
    kNearestNeighbor(kNearestNeighbor<T, Distance> &&inputObj) : drows(inputObj.drows),
                                                                 qrows(inputObj.qrows),
                                                                 cols(inputObj.cols),
                                                                 nn(inputObj.nn),
                                                                 indices_ptr(std::move(inputObj.indices_ptr)),
                                                                 dists_ptr(std::move(inputObj.dists_ptr)),
                                                                 indices(std::move(inputObj.indices)),
                                                                 dists(std::move(inputObj.dists)),
                                                                 the_same(inputObj.the_same)
    {
    }

    /*!
     * \brief Copy constructor
     * \param inputObj kNearestNeighbor to be copied
     */
    kNearestNeighbor(kNearestNeighbor<T, Distance> const &inputObj) : drows(inputObj.drows),
                                                                      qrows(inputObj.qrows),
                                                                      cols(inputObj.cols),
                                                                      nn(inputObj.nn),
                                                                      indices_ptr(new int[inputObj.qrows * inputObj.nn]),
                                                                      dists_ptr(new T[inputObj.qrows * inputObj.nn]),
                                                                      indices(indices_ptr.get(), inputObj.qrows, inputObj.nn),
                                                                      dists(dists_ptr.get(), inputObj.qrows, inputObj.nn),
                                                                      the_same(inputObj.the_same)
    {
        {
            int *From = inputObj.indices_ptr.get();
            int *To = indices_ptr.get();
            std::copy(From, From + qrows * nn, To);
        }
        {
            T *From = inputObj.dists_ptr.get();
            T *To = dists_ptr.get();
            std::copy(From, From + qrows * nn, To);
        }
    }

    /*!
     * \brief Move assignment operator
     * \param inputObj kNearestNeighbor to be assigned
     */
    kNearestNeighbor<T, Distance> &operator=(kNearestNeighbor<T, Distance> &&inputObj)
    {
        drows = std::move(inputObj.drows);
        qrows = std::move(inputObj.qrows);
        cols = std::move(inputObj.cols);
        nn = std::move(inputObj.nn);
        the_same = std::move(inputObj.the_same);
        indices_ptr = std::move(inputObj.indices_ptr);
        dists_ptr = std::move(inputObj.dists_ptr);
        indices = std::move(inputObj.indices);
        dists = std::move(inputObj.dists);
        return *this;
    }

    /*!
     * \brief Default destructor
     *
     */
    ~kNearestNeighbor() {}

    /*!
     * \brief Construct a kd-tree index & do a knn search
     * 
     * \param idata A pointer to input data 
     */
    void buildIndex(T *idata)
    {
        flann::Matrix<T> dataset(idata, drows, cols);

        //Construct an randomized kd-tree index using 4 kd-trees
        //For the number of parallel kd-trees to use (Good values are in the range [1..16])
        flann::Index<Distance> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        //Do a knn search, using 128 checks
        //Number of checks means: How many leafs to visit when searching
        //for neighbours (-1 for unlimited)
        index.knnSearch(dataset, indices, dists, nn, flann::SearchParams(128));
    }

    /*!
     * \brief Construct a kd-tree index & do a knn search
     * 
     * \param idata A pointer to input data 
     * \param qdata A pointer to query data 
     */
    void buildIndex(T *idata, T *qdata)
    {
        flann::Matrix<T> dataset(idata, drows, cols);

        //Construct an randomized kd-tree index using 4 kd-trees
        //For the number of parallel kd-trees to use (Good values are in the range [1..16])
        flann::Index<Distance> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        flann::Matrix<T> query(qdata, qrows, cols);

        //Do a knn search, using 128 checks
        //Number of checks means: How many leafs to visit when searching
        //for neighbours (-1 for unlimited)
        index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

        if (!checkNearestNeighbors())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Input data & query data are the same!" << std::endl;
        }
    }

    /*!
     * \brief A pointer to nearest neighbors indices
     * 
     * \param index Index of a point (from data points) to get its neighbors
     * 
     * \returns A (pointer to a) row of the nearest neighbors indices.
     */
    inline int *NearestNeighbors(int const &index) const
    {
        //+1 is that we do not want the index of the point itself
        return indices_ptr.get() + index * nn + the_same;
    }

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
    inline int *NearestNeighbors() const
    {
        return indices_ptr.get();
    }

    /*!
     * \brief A pointer to nearest neighbors distances from the point index
     * 
     * \param index Index of a point (from data points) 
     * 
     * \returns A pointer to nearest neighbors distances from the point index
     */
    inline T *NearestNeighborsDistances(int const &index) const
    {
        //+1 is that we do not want the index of the point itself
        return dists_ptr.get() + index * nn + the_same;
    }

    /*!
     * \brief Distance of a nearest neighbor of index
     * 
     * \param index Index of a point (from data points) 
     * 
     * \returns Distance of a nearest neighbor point of the index
     */
    inline T minDist(int const &index) const
    {
        std::ptrdiff_t const Id = index * nn + the_same;
        return dists_ptr[Id];
    }

    /*!
     * \brief Vector of all points' distance of their nearest neighbor 
     * 
     * \returns Vector of all points' distance of their nearest neighbor 
     */
    inline T *minDist()
    {
        T *mindists = nullptr;
        try
        {
            mindists = new T[qrows];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return nullptr;
        }

        for (std::size_t i = 0; i < qrows; ++i)
        {
            std::ptrdiff_t const Id = i * nn + the_same;
            mindists[i] = dists_ptr[Id];
        }

        return mindists;
    }

    /*!
     * \brief  Nmber of each point nearest neighbors
     * 
     * \returns number of nearest neighbors
     */
    inline int numNearestNeighbors() const
    {
        return nn - the_same;
    }

    /*!
     * \brief   Function to make sure that we do not compute the nearest neighbors of a point from itself
     * 
     * \returns true for if input points and query points are used correctly
     */
    bool checkNearestNeighbors()
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

    /*!
     * \brief swap two indexes values
     */
    void IndexSwap(int Indx1, int Indx2)
    {
        std::swap(indices_ptr[Indx1], indices_ptr[Indx2]);
        std::swap(dists_ptr[Indx1], dists_ptr[Indx2]);
    }

    /*!
     * \returns number of input data points
     */
    inline int numInputdata() const
    {
        return drows;
    }

    /*!
     * \returns number of query data points
     */
    inline int numQuerydata() const
    {
        return qrows;
    }

  private:
    //! Number of data rows
    std::size_t drows;

    //! Number of qury rows
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

//TODO : Somehow the specialized template did not work.
//FIXME: to the correct templated version

/*! \class L2NearestNeighbor
 * \brief Finding K nearest neighbors in high dimensional spaces using L2 distance functor
 * 
 * \tparam T data type
 */
template <typename T>
class L2NearestNeighbor : public kNearestNeighbor<T, flann::L2<T>>
{
  public:
    L2NearestNeighbor(int const ndataPoints, int const nDim, int const nN) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nDim, nN)
    {
    }
    L2NearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const nN) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nqueryPoints, nDim, nN) {}
    L2NearestNeighbor(L2NearestNeighbor<T> &&inputObj) : kNearestNeighbor<T, flann::L2<T>>(std::move(inputObj)) {}
    L2NearestNeighbor(L2NearestNeighbor<T> const &inputObj) : kNearestNeighbor<T, flann::L2<T>>(inputObj) {}
    L2NearestNeighbor<T> &operator=(L2NearestNeighbor<T> &&inputObj)
    {
        kNearestNeighbor<T, flann::L2<T>>::drows = std::move(inputObj.drows);
        kNearestNeighbor<T, flann::L2<T>>::qrows = std::move(inputObj.qrows);
        kNearestNeighbor<T, flann::L2<T>>::cols = std::move(inputObj.cols);
        kNearestNeighbor<T, flann::L2<T>>::nn = std::move(inputObj.nn);
        kNearestNeighbor<T, flann::L2<T>>::the_same = std::move(inputObj.the_same);
        kNearestNeighbor<T, flann::L2<T>>::indices_ptr = std::move(inputObj.indices_ptr);
        kNearestNeighbor<T, flann::L2<T>>::dists_ptr = std::move(inputObj.dists_ptr);
        kNearestNeighbor<T, flann::L2<T>>::indices = std::move(inputObj.indices);
        kNearestNeighbor<T, flann::L2<T>>::dists = std::move(inputObj.dists);
        return static_cast<L2NearestNeighbor<T> &>(kNearestNeighbor<T, flann::L2<T>>::operator=(std::move(inputObj)));
    }
};

// namespace flann
// {
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

#endif //UMUQ_FLANNLIB_H
