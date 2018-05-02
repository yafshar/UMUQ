#ifndef UMHBM_FLANNLIB_H
#define UMHBM_FLANNLIB_H

#if HAVE_FLANN
#include <flann/flann.hpp>

template <typename T>
class aNearestNeighbor
{
  public:
    aNearestNeighbor() {}
    ~aNearestNeighbor() {}

//     Matrix<int> indices(new int[query.rows * nn], query.rows, nn);
//     Matrix<T> dists(new T[query.rows * nn], query.rows, nn);

//     //Construct an randomized kd-tree index using 4 kd-trees
//     //For the number of parallel kd-trees to use (Good values are in the range [1..16])
//     flann::Index<L2<T>> index(dataset, flann::KDTreeIndexParams(4));
//     index.buildIndex();

//     // do a knn search, using 128 checks
//     index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

//     flann::save_to_file(indices, "result.hdf5", "result");

//     delete[] dataset.ptr();
//     delete[] query.ptr();
//     delete[] indices.ptr();
//     delete[] dists.ptr();

//   private:
//     flann::Matrix<T> m;
//     flann::Matrix<T> q;

//     flann::Matrix<int> indices;
//     flann::Matrix<T> dists;

//     std::size_t rows;
//     std::size_t cols;
//     std::size_t stride;

//     //how many nearest neighbors to find
//     int nn;

//     enum class
//     {
//         "EUCLIDEAN",
//         "L2",
//         "MANHATTAN",
//         "L1",
//         "MINKOWSKI",
//         "MAX",
//         "HIST_INTERSECT",
//         "HELLINGER",
//         "CHI_SQUARE",
//         "KULLBACK_LEIBLER",
//         "HAMMING",
//         "HAMMING_LUT",
//         "HAMMING_POPCNT",
//         "L2_SIMPLE",
//     };
};

#endif //HAVE_FLANN
#endif //
