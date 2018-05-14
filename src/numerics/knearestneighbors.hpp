#ifndef UMUQ_KNEARESTNEIGHBORS_H
#define UMUQ_KNEARESTNEIGHBORS_H

#ifdef HAVE_FLANN
/*!
 * FLANN is a library for performing fast approximate nearest neighbor searches in high dimensional spaces. 
 * It contains a collection of algorithms we found to work best for nearest neighbor search and a system 
 * for automatically choosing the best algorithm and optimum parameters depending on the dataset.
 */
#include <flann/flann.hpp>
#endif

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
 * \b MINKOWSKI
 * \b MAX
 * \b HIST_INTERSECT
 * \b HELLINGER
 * \b CHI_SQUARE
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
#ifdef HAVE_FLANN
                                                                            indices(indices_ptr.get(), ndataPoints, (nN + 1)),
                                                                            dists(dists_ptr.get(), ndataPoints, (nN + 1)),
#endif
                                                                            the_same(true)
    {
    }

    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const nN) : drows(ndataPoints),
                                                                                                    qrows(nqueryPoints),
                                                                                                    cols(nDim),
                                                                                                    nn(nN),
                                                                                                    indices_ptr(new int[nqueryPoints * nN]),
                                                                                                    dists_ptr(new T[nqueryPoints * nN]),
#ifdef HAVE_FLANN
                                                                                                    indices(indices_ptr.get(), nqueryPoints, nN),
                                                                                                    dists(dists_ptr.get(), nqueryPoints, nN),
#endif
                                                                                                    the_same(false)
    {
    }

    /*!
     * Move constructor
     * \param inputObj kNearestNeighbor to be moved
     */
    kNearestNeighbor(kNearestNeighbor &&inputObj)
    {
        //Number of data rows
        drows = inputObj.drows;
        //Number of qury rows
        qrows = inputObj.qrows;
        //Number of columns
        cols = inputObj.cols;
        //Number of nearest neighbors to find
        nn = inputObj.nn;
        //Flag to check if the input data and qury data are the same
        the_same = inputObj.the_same;

        indices_ptr = std::move(inputObj.indices_ptr);
        dists_ptr = std::move(inputObj.dists_ptr);

#ifdef HAVE_FLANN
        indices = std::move(inputObj.indices);
        dists = std::move(inputObj.dists);
#endif
    }

    /*!
     * Move assignment operator
     * \param inputObj kNearestNeighbor to be assigned
     */
    kNearestNeighbor &operator=(kNearestNeighbor &&inputObj)
    {
        //Number of data rows
        drows = inputObj.drows;
        //Number of qury rows
        qrows = inputObj.qrows;
        //Number of columns
        cols = inputObj.cols;
        //Number of nearest neighbors to find
        nn = inputObj.nn;
        //Flag to check if the input data and qury data are the same
        the_same = inputObj.the_same;

        indices_ptr = std::move(inputObj.indices_ptr);
        dists_ptr = std::move(inputObj.dists_ptr);

#ifdef HAVE_FLANN
        indices = std::move(inputObj.indices);
        dists = std::move(inputObj.dists);
#endif
        return *this;
    }

    /*!
     * \brief destructor
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
#ifdef HAVE_FLANN
        flann::Matrix<T> dataset(idata, drows, cols);

        //Construct an randomized kd-tree index using 4 kd-trees
        //For the number of parallel kd-trees to use (Good values are in the range [1..16])
        flann::Index<Distance> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        //Do a knn search, using 128 checks
        //Number of checks means: How many leafs to visit when searching
        //for neighbours (-1 for unlimited)
        index.knnSearch(dataset, indices, dists, nn, flann::SearchParams(128));
#endif
    }

    /*!
     * \brief Construct a kd-tree index & do a knn search
     * 
     * \param idata A pointer to input data 
     * \param qdata A pointer to query data 
     */
    void buildIndex(T *idata, T *qdata)
    {
#ifdef HAVE_FLANN
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
#endif
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
        T *dists = nullptr;
        try
        {
            dists = new T[qrows];
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
            dists[i] = dists_ptr[Id];
        }

        return dists;
    }

    /*!
     * \brief  Nmber of each point nearest neighbors
     * 
     * \returns number of nearest neighbors
     */
    inline int const numNearestNeighbors() const
    {
        return nn - the_same;
    }





  private:
    std::unique_ptr<int[]> indices_ptr;
    std::unique_ptr<T[]> dists_ptr;

#ifdef HAVE_FLANN
    flann::Matrix<int> indices;
    flann::Matrix<T> dists;
#endif

    //Number of data rows
    std::size_t drows;

    //Number of qury rows
    std::size_t qrows;

    //Number of columns
    std::size_t cols;

    //Number of nearest neighbors to find
    int nn;

    //Flag to check if the input data and qury data are the same
    bool the_same;
};

//TODO : Somehow the specialized template did not work.
//FIXME: to the correct templated version

/*! \class L2NearestNeighbor
 * \brief Finding K nearest neighbors in high dimensional spaces using L2 distance functor
 * 
 * \tparam T  data type
 */
template <typename T>
#ifdef HAVE_FLANN
class L2NearestNeighbor : public kNearestNeighbor<T, flann::L2<T>>
#else
class L2NearestNeighbor : public kNearestNeighbor<T, T>
#endif
{
  public:
#ifdef HAVE_FLANN
    L2NearestNeighbor(int const ndataPoints, int const nDim, int const nN) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nDim, nN)
    {
    }
    L2NearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const nN) : kNearestNeighbor<T, flann::L2<T>>(ndataPoints, nqueryPoints, nDim, nN) {}
#else
    L2NearestNeighbor(int const ndataPoints, int const nDim, int const nN) : kNearestNeighbor<T, T>(ndataPoints, nDim, nN)
    {
    }
    L2NearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const nN) : kNearestNeighbor<T, T>(ndataPoints, nqueryPoints, nDim, nN) {}
#endif
};

#endif //UMUQ_FLANNLIB_H
