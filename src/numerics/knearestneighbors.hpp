#ifndef UMHBM_KNEARESTNEIGHBORS_H
#define UMHBM_KNEARESTNEIGHBORS_H

#if HAVE_FLANN
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
 * \b EUCLIDEAN
 * \b L2             Squared Euclidean distance functor, optimized version 
 * \b MANHATTAN
 * \b L1             Manhattan distance functor, optimized version
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
 * \b L2_SIMPLE      Squared Euclidean distance functor
 */
template <typename T, class Distance>
class kNearestNeighbor
{
  public:
    /*!
     * \brief constructor
     * 
     * \param nPoints Number of data points
     * \param nDim    Dimension of each point
     * \param nN      Number of nearest neighbors to find
     */
    kNearestNeighbor(int const nPoints, int const nDim, int const nN) : rows(nPoints),
                                                                        cols(nDim),
                                                                        nn(nN + 1),
                                                                        indices_ptr(new int[nPoints * (nN + 1)]),
                                                                        dists_ptr(new T[nPoints * (nN + 1)]),
                                                                        indices(indices_ptr.get(), nPoints, (nN + 1)),
                                                                        dists(dists_ptr.get(), nPoints, (nN + 1)) {}

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
        flann::Matrix<T> dataset(idata, rows, cols);

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
     * \brief A pointer to nearest neighbors indices
     * 
     * \param index Index of a point (from data points) to get its neighbors
     * 
     * \returns A (pointer to a) row of the nearest neighbors indices.
     */
    inline int *NearestNeighbors(int const &index) const
    {
        //+1 is that we do not want the index of the point itself
        return indices_ptr.get() + index * nn + 1;
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
     *  Each row has the size of nn which is the number of neighbors+1
     *  and it has nPoints columns
     * 
     * \returns All points nearest neighbors indices (in row porder).
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
        return dists_ptr.get() + index * nn + 1;
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
        std::ptrdiff_t const Id = index * nn + 1;
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
            dists = new T[rows];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return nullptr;
        }

        for (std::size_t i = 0; i < rows; ++i)
        {
            std::ptrdiff_t const Id = i * nn + 1;
            dists[i] = dists_ptr[Id];
        }

        return dists;
    }

  private:
    std::unique_ptr<int[]> indices_ptr;
    std::unique_ptr<T[]> dists_ptr;

    flann::Matrix<int> indices;
    flann::Matrix<T> dists;

    std::size_t rows;
    std::size_t cols;

    //Number of nearest neighbors to find
    int nn;
};

//TODO : Somehow the specialized template did not work.
//FIXME: to the correct templated version

/*! \class L2NearestNeighbor
 * \brief Finding K nearest neighbors in high dimensional spaces using L2 distance functor
 * 
 * \tparam T  data type
 */
template <typename T>
class L2NearestNeighbor : public kNearestNeighbor<T, flann::L2<T>>
{
  public:
    L2NearestNeighbor(int const nPoints, int const nDim, int const nN) : kNearestNeighbor<T, flann::L2<T>>(nPoints, nDim, nN) {}
};

#endif //HAVE_FLANN
#endif //UMHBM_FLANNLIB_H
