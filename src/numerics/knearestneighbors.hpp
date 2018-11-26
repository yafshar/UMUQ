#ifndef UMUQ_KNEARESTNEIGHBORS_H
#define UMUQ_KNEARESTNEIGHBORS_H

#include "knearestneighborbase.hpp"

namespace umuq
{

/*! \enum NeighborDistance
 * \ingroup Numerics_Module
 * 
 * \brief Nearest neighbors distance types which can be used in %UMUQ
 * 
 */
enum class NeighborDistance
{
    /*! Squared Euclidean distance functor, optimized version. */
    EUCLIDEAN,
    /*! Squared Euclidean distance functor, optimized version. */
    L2 = EUCLIDEAN,
    /*! Squared Euclidean distance functor. An unrolled simpler version is used which is preferable for very low dimensionality data (eg 3D points) */
    L2_SIMPLE,
    /*! Manhattan distance functor, optimized version. */
    MANHATTAN,
    /*! Manhattan distance functor, optimized version. */
    L1 = MANHATTAN,
    /*! The Minkowsky \f$ (L_p) (\vec{x},\vec{y}) = \left( \sum_{i = 1}^{n} | x_i - y_i |^p \right)^{\frac{1}{p}} \f$ distance between two vectors \f$ \vec{x}~\text{and}~\vec{x}. \f$ */
    MINKOWSKI,
    /*! Maximum distance.*/
    MAX,
    /*! The histogram intersection distance. */
    HIST_INTERSECT,
    /*! The Hellinger distance, quantify the similarity between two probability distributions. */
    HELLINGER,
    /*! The distance between two histograms. */
    CHI_SQUARE,
    /*! Kullback-Leibler divergence, is a measure of how one probability distribution is different from a second, reference probability distribution.*/
    KULLBACK_LEIBLER,
    /*! Hamming distance functor - counts the number of positions at which the corresponding symbols are different. */
    HAMMING,
    /*! Hamming distance functor - counts the bit differences between two strings, useful for the brief descriptor bit count of A exclusive \c XOR'ed with B */
    HAMMING_LUT,
    /*! Hamming distance functor (pop count between two binary vectors, i.e. xor them and count the number of bits set). */
    HAMMING_POPCNT,
    /*! The Mahalanobis distance is a measure of the distance between a point and a distribution. */
    MAHALANOBIS
};

/*! \class kNearestNeighbor
 * \ingroup Numerics_Module
 * 
 * \brief Finding K nearest neighbors in high dimensional spaces
 * 
 * \tparam DataType      Data type
 * \tparam DistanceType  NeighborDistance type for computing the distances to the nearest neighbors.
 *                       (Default is EUCLIDEAN distance) 
 *                       \sa umuq::NeighborDistance
 */
template <typename DataType, NeighborDistance DistanceType = NeighborDistance::EUCLIDEAN>
class kNearestNeighbor : public kNearestNeighborBase<DataType, flann::L2<DataType>>
{
  public:
    /*!
     * \brief Construct a new kNearestNeighbor object
     *
     * \param ndataPoints  Number of data points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);

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
     * \brief Move construct a new kNearestNeighbor object
     *
     * \param other kNearestNeighbor object
     */
    kNearestNeighbor(kNearestNeighbor<DataType, DistanceType> &&other);

    /*!
     * \brief Copy construct a new kNearestNeighbor object
     *
     * \param other kNearestNeighbor object
     */
    kNearestNeighbor(kNearestNeighbor<DataType, DistanceType> const &other);

    /*!
     * \brief Move assignment
     *
     * \param other kNearestNeighbor object
     * \returns kNearestNeighbor<DataType>&
     */
    kNearestNeighbor<DataType, DistanceType> &operator=(kNearestNeighbor<DataType, DistanceType> &&other);

  private:
    /*!
     * \brief Explicitly prevent the default construct a new k Nearest Neighbor object
     * 
     */
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> : public kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> const &other);
    kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> &operator=(kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> : public kNearestNeighborBase<DataType, flann::L1<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> const &other);
    kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> &operator=(kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> : public kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> const &other);
    kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> &operator=(kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::MAX> : public kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MAX> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MAX> const &other);
    kNearestNeighbor<DataType, NeighborDistance::MAX> &operator=(kNearestNeighbor<DataType, NeighborDistance::MAX> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> : public kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> const &other);
    kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> &operator=(kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::HELLINGER> : public kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HELLINGER> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HELLINGER> const &other);
    kNearestNeighbor<DataType, NeighborDistance::HELLINGER> &operator=(kNearestNeighbor<DataType, NeighborDistance::HELLINGER> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> : public kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> const &other);
    kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> &operator=(kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> &&other);

  private:
    kNearestNeighbor() = delete;
};
template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> : public kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> const &other);
    kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> &operator=(kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> &&other);

  private:
    kNearestNeighbor() = delete;
};
template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::HAMMING> : public kNearestNeighborBase<DataType, flann::Hamming<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING> const &other);
    kNearestNeighbor<DataType, NeighborDistance::HAMMING> &operator=(kNearestNeighbor<DataType, NeighborDistance::HAMMING> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> : public kNearestNeighborBase<DataType, flann::HammingLUT>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> const &other);
    kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> &operator=(kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> : public kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> const &other);
    kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> &operator=(kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> &&other);

  private:
    kNearestNeighbor() = delete;
};

/*! \class kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>
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
 * \tparam DataType Data type
 */
template <typename DataType>
class kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> : public kNearestNeighborBase<DataType, flann::L2<DataType>>
{
  public:
    /*!
     * \brief Construct a new kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> object
     *
     * \param ndataPoints  Number of data points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Construct a new kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> object
     *
     * \param ndataPoints  Number of data points
     * \param nqueryPoints Number of query points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Move construct a new kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> object
     *
     * \param other kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> object
     */
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> &&other);

    /*!
     * \brief Copy construct a new kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> object
     *
     * \param other kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> object
     */
    kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> const &other);

    /*!
     * \brief Move assignment
     *
     * \param other kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> object
     * \returns kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>&
     */
    kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> &operator=(kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> &&other);

    /*!
     * \brief Construct a kd-tree index & do a knn search
     *
     * \param idata A pointer to input data
     */
    void buildIndex(DataType *idata);

    /*!
     * \brief Construct a kd-tree index & do a knn search
     *
     * \param idata A pointer to input data
     * \param qdata A pointer to query data
     */
    void buildIndex(DataType *idata, DataType *qdata);

    /*!
     * \brief Set the Covariance object
     *
     * \param Covariance  The covariance matrix
     */
    inline void setCovariance(EMatrixX<DataType> const &Covariance);

    /*!
     * \brief Set the Covariance object
     *
     * \param Covariance  The covariance matrix
     */
    inline void setCovariance(DataType const *Covariance);

    /*!
     * \brief Set the Covariance object
     *
     * \param Covariance  The covariance matrix
     */
    inline void setCovariance(std::vector<DataType> const &Covariance);

    /*!
     * \brief Access the covariance matrix.
     *
     * \returns Constant reference to the covariance matrix.
     */
    inline EMatrixX<DataType> const &Covariance() const;

    /*!
     * \brief Modify the covariance matrix.
     *
     * \returns Reference to the covariance matrix.
     */
    inline EMatrixX<DataType> &Covariance();

  private:
    kNearestNeighbor() = delete;

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
     */
    EMatrixX<DataType> covariance;

    /*! A view of the lower triangular matrix L from the Cholesky decomposition of inverse covariance matrix */
    EMatrixX<DataType> matrixL;
};

template <typename DataType, NeighborDistance DistanceType>
kNearestNeighbor<DataType, DistanceType>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType, NeighborDistance DistanceType>
kNearestNeighbor<DataType, DistanceType>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType, NeighborDistance DistanceType>
kNearestNeighbor<DataType, DistanceType>::kNearestNeighbor(kNearestNeighbor<DataType, DistanceType> &&other) : kNearestNeighborBase<DataType, flann::L2<DataType>>(std::move(other))
{
}

template <typename DataType, NeighborDistance DistanceType>
kNearestNeighbor<DataType, DistanceType>::kNearestNeighbor(kNearestNeighbor<DataType, DistanceType> const &other) : kNearestNeighborBase<DataType, flann::L2<DataType>>(other)
{
}

template <typename DataType, NeighborDistance DistanceType>
kNearestNeighbor<DataType, DistanceType> &kNearestNeighbor<DataType, DistanceType>::operator=(kNearestNeighbor<DataType, DistanceType> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, DistanceType> &>(kNearestNeighborBase<DataType, flann::L2<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> &&other) : kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> const &other) : kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> &kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE>::operator=(kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::L2_SIMPLE> &>(kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MANHATTAN>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L1<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MANHATTAN>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L1<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MANHATTAN>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> &&other) : kNearestNeighborBase<DataType, flann::L1<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MANHATTAN>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> const &other) : kNearestNeighborBase<DataType, flann::L1<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> &kNearestNeighbor<DataType, NeighborDistance::MANHATTAN>::operator=(kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::MANHATTAN> &>(kNearestNeighborBase<DataType, flann::L1<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> &&other) : kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> const &other) : kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> &kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI>::operator=(kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::MINKOWSKI> &>(kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAX>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAX>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAX>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MAX> &&other) : kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAX>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MAX> const &other) : kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAX> &kNearestNeighbor<DataType, NeighborDistance::MAX>::operator=(kNearestNeighbor<DataType, NeighborDistance::MAX> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::MAX> &>(kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> &&other) : kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> const &other) : kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> &kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT>::operator=(kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::HIST_INTERSECT> &>(kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HELLINGER>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HELLINGER>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HELLINGER>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HELLINGER> &&other) : kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HELLINGER>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HELLINGER> const &other) : kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HELLINGER> &kNearestNeighbor<DataType, NeighborDistance::HELLINGER>::operator=(kNearestNeighbor<DataType, NeighborDistance::HELLINGER> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::HELLINGER> &>(kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> &&other) : kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> const &other) : kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> &kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE>::operator=(kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::CHI_SQUARE> &>(kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> &&other) : kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> const &other) : kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> &kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER>::operator=(kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::KULLBACK_LEIBLER> &>(kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::Hamming<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::Hamming<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING> &&other) : kNearestNeighborBase<DataType, flann::Hamming<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING> const &other) : kNearestNeighborBase<DataType, flann::Hamming<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING> &kNearestNeighbor<DataType, NeighborDistance::HAMMING>::operator=(kNearestNeighbor<DataType, NeighborDistance::HAMMING> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::HAMMING> &>(kNearestNeighborBase<DataType, flann::Hamming<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HammingLUT>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HammingLUT>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> &&other) : kNearestNeighborBase<DataType, flann::HammingLUT>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> const &other) : kNearestNeighborBase<DataType, flann::HammingLUT>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> &kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT>::operator=(kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::HAMMING_LUT> &>(kNearestNeighborBase<DataType, flann::HammingLUT>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> &&other) : kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> const &other) : kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> &kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT>::operator=(kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::HAMMING_POPCNT> &>(kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2<DataType>>(ndataPoints, nDim, kNeighbors),
                                                                                                                                           covariance(EMatrixX<DataType>::Identity(nDim, nDim)),
                                                                                                                                           matrixL(EMatrixX<DataType>::Identity(nDim, nDim))
{
    if (!(unrolledIncrement == 0 || unrolledIncrement == 4 || unrolledIncrement == 6 || unrolledIncrement == 8 || unrolledIncrement == 10 || unrolledIncrement == 12))
    {
        UMUQFAIL("The unrolled increment value of unrolledIncrement is not correctly set!");
    }
    this->withCovariance = true;
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors),
                                                                                                                                                                   covariance(EMatrixX<DataType>::Identity(nDim, nDim)),
                                                                                                                                                                   matrixL(EMatrixX<DataType>::Identity(nDim, nDim))
{
    if (!(unrolledIncrement == 0 || unrolledIncrement == 4 || unrolledIncrement == 6 || unrolledIncrement == 8 || unrolledIncrement == 10 || unrolledIncrement == 12))
    {
        UMUQFAIL("The unrolled increment value of unrolledIncrement is not correctly set!");
    }
    this->withCovariance = true;
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> &&other) : kNearestNeighborBase<DataType, flann::L2<DataType>>(std::move(other))
{
    this->covariance = std::move(other.covariance);
    this->matrixL = std::move(other.matrixL);
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::kNearestNeighbor(kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> const &other) : kNearestNeighborBase<DataType, flann::L2<DataType>>(other)
{
    this->covariance = other.covariance;
    this->matrixL = other.matrixL;
}

template <typename DataType>
kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> &kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::operator=(kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> &&other)
{
    this->nDataPoints = std::move(other.nDataPoints);
    this->nQueryDataPoints = std::move(other.nQueryDataPoints);
    this->dataDimension = std::move(other.dataDimension);
    this->nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
    this->indices_ptr = std::move(other.indices_ptr);
    this->dists_ptr = std::move(other.dists_ptr);
    this->indices = std::move(other.indices);
    this->dists = std::move(other.dists);
    this->the_same = std::move(other.the_same);
    this->withCovariance = std::move(other.withCovariance);
    this->covariance = std::move(other.covariance);
    this->matrixL = std::move(other.matrixL);

    return static_cast<kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS> &>(kNearestNeighborBase<DataType, flann::L2<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
void kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::buildIndex(DataType *idata)
{
    if (covariance.isIdentity(1e-3))
    {
        kNearestNeighborBase<DataType, flann::L2<DataType>>::buildIndex(idata);
    }
    else
    {
        // Copy the input data into a temporary array
        std::vector<DataType> inputData{idata, idata + this->nDataPoints * this->dataDimension};

        // Map the temporary array in Eigen format without memory copy
        EMapType<DataType> inputDataEMap(inputData.data(), this->nDataPoints, this->dataDimension);

        // Compute the input data matrix multiply by lower triangular matrix L from
        // the Cholesky decomposition of inverse covariance matrix
        inputDataEMap *= matrixL;

        // Map the data in flann matrix format
        flann::Matrix<DataType> dataset(inputData.data(), this->nDataPoints, this->dataDimension);

        // Construct an randomized kd-tree index using 4 kd-trees
        // For the number of parallel kd-trees to use (Good values are in the range [1..16])
        flann::Index<flann::L2<DataType>> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        // Do a knn search, using 128 checks
        // Number of checks means: How many leafs to visit when searching
        // for neighbors (-1 for unlimited)
        index.knnSearch(dataset, this->indices, this->dists, this->nNearestNeighborsToFind, flann::SearchParams(128));

        // Total number of nearest neighbors for each point
        int nNN = this->numNearestNeighbors();

        // Correct the distances
        // Loop over all points
        for (int i = 0; i < this->nDataPoints; i++)
        {
            std::ptrdiff_t const IdI = i * this->dataDimension;

            // A pointer to nearest neighbors indices of point i
            int *NearestNeighbors = this->NearestNeighbors(i);

            // A pointer to nearest neighbors square distances from the point i
            DataType *nnDist = this->NearestNeighborsDistances(i);

#if unrolledIncrement == 0
            {
                DataType *last = idata + IdI + this->dataDimension;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->dataDimension;

                    // pointer to query data
                    DataType *Idata = idata + IdI;

                    // pointer to idata (neighbors of i)
                    DataType *Jdata = idata + IdJ;

                    DataType result(0);
                    while (Idata < last)
                    {
                        DataType const diff0 = *Idata++ - *Jdata++;
                        result += diff0 * diff0;
                    }

                    nnDist[j] = result;
                }
            }
#else
            {
                DataType *last = idata + IdI + this->dataDimension;
                DataType *lastgroup = last - unrolledIncrement + 1;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->dataDimension;

                    // pointer to query data
                    DataType *Idata = idata + IdI;

                    // pointer to idata (neighbors of i)
                    DataType *Jdata = idata + IdJ;

                    DataType result(0);

                    // Process unrolled Increment items with each loop for efficiency.
                    while (Idata < lastgroup)
                    {
                        DataType const diff0 = Idata[0] - Jdata[0];
                        DataType const diff1 = Idata[1] - Jdata[1];
                        DataType const diff2 = Idata[2] - Jdata[2];
                        DataType const diff3 = Idata[3] - Jdata[3];
#if unrolledIncrement == 4
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
#endif
#if unrolledIncrement == 6
                        DataType const diff4 = Idata[4] - Jdata[4];
                        DataType const diff5 = Idata[5] - Jdata[5];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5;
#endif
#if unrolledIncrement == 8
                        DataType const diff4 = Idata[4] - Jdata[4];
                        DataType const diff5 = Idata[5] - Jdata[5];
                        DataType const diff6 = Idata[6] - Jdata[6];
                        DataType const diff7 = Idata[7] - Jdata[7];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
#endif
#if unrolledIncrement == 10
                        DataType const diff4 = Idata[4] - Jdata[4];
                        DataType const diff5 = Idata[5] - Jdata[5];
                        DataType const diff6 = Idata[6] - Jdata[6];
                        DataType const diff7 = Idata[7] - Jdata[7];
                        DataType const diff8 = Idata[8] - Jdata[8];
                        DataType const diff9 = Idata[9] - Jdata[9];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9;
#endif
#if unrolledIncrement == 12
                        DataType const diff4 = Idata[4] - Jdata[4];
                        DataType const diff5 = Idata[5] - Jdata[5];
                        DataType const diff6 = Idata[6] - Jdata[6];
                        DataType const diff7 = Idata[7] - Jdata[7];
                        DataType const diff8 = Idata[8] - Jdata[8];
                        DataType const diff9 = Idata[9] - Jdata[9];
                        DataType const diff10 = Idata[10] - Jdata[10];
                        DataType const diff11 = Idata[11] - Jdata[11];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9 + diff10 * diff10 + diff11 * diff11;
#endif
                        Idata += unrolledIncrement;
                        Jdata += unrolledIncrement;
                    }
                    // Process last pixels.
                    while (Idata < last)
                    {
                        DataType const diff0 = *Idata++ - *Jdata++;
                        result += diff0 * diff0;
                    }

                    nnDist[j] = result;
                }
            }
#endif
        }
    }
}

template <typename DataType>
void kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::buildIndex(DataType *idata, DataType *qdata)
{
    if (covariance.isIdentity(1e-3))
    {
        kNearestNeighborBase<DataType, flann::L2<DataType>>::buildIndex(idata, qdata);
    }
    else
    {
        // Copy the input data into a temporary array
        std::vector<DataType> inputData{idata, idata + this->nDataPoints * this->dataDimension};

        // Copy the query data into a temporary array
        std::vector<DataType> queryData{qdata, qdata + this->nQueryDataPoints * this->dataDimension};

        // Map the temporary array in Eigen format without memory copy
        EMapType<DataType> inputDataEMap(inputData.data(), this->nDataPoints, this->dataDimension);

        // Map the temporary array in Eigen format without memory copy
        EMapType<DataType> queryDataEMap(queryData.data(), this->nQueryDataPoints, this->dataDimension);

        // Compute the input data matrix multiply by lower triangular matrix L from
        // the Cholesky decomposition of inverse covariance matrix
        inputDataEMap *= matrixL;

        queryDataEMap *= matrixL;

        // Map the data in flann matrix format
        flann::Matrix<DataType> dataset(inputData.data(), this->nDataPoints, this->dataDimension);

        // Construct an randomized kd-tree index using 4 kd-trees
        // For the number of parallel kd-trees to use (Good values are in the range [1..16])
        flann::Index<flann::L2<DataType>> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        flann::Matrix<DataType> query(queryData.data(), this->nQueryDataPoints, this->dataDimension);

        // Do a knn search, using 128 checks
        // Number of checks means: How many leafs to visit when searching
        // for neighbors (-1 for unlimited)
        index.knnSearch(query, this->indices, this->dists, this->nNearestNeighborsToFind, flann::SearchParams(128));

        if (!this->checkNearestNeighbors())
        {
            UMUQWARNING("Input data & query data are the same!");
        }

        // Total number of nearest neighbors for each point
        int const nNN = this->numNearestNeighbors();

        // Correct the distances
        // Loop over all query points
        for (int i = 0; i < this->nQueryDataPoints; i++)
        {
            std::ptrdiff_t const IdI = i * this->dataDimension;

            // A pointer to nearest neighbors indices of point i
            int *NearestNeighbors = this->NearestNeighbors(i);

            // A pointer to nearest neighbors square distances from the point i
            DataType *nnDist = this->NearestNeighborsDistances(i);

#if unrolledIncrement == 0
            {
                DataType *last = qdata + IdI + this->dataDimension;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->dataDimension;

                    // pointer to query data
                    DataType *Idata = qdata + IdI;

                    // pointer to idata (neighbors of i)
                    DataType *Jdata = idata + IdJ;

                    DataType result(0);
                    while (Idata < last)
                    {
                        DataType const diff0 = *Idata++ - *Jdata++;
                        result += diff0 * diff0;
                    }

                    nnDist[j] = result;
                }
            }
#else
            {
                DataType *last = qdata + IdI + this->dataDimension;
                DataType *lastgroup = last - unrolledIncrement + 1;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->dataDimension;

                    // pointer to query data
                    DataType *Idata = qdata + IdI;

                    // pointer to idata (neighbors of i)
                    DataType *Jdata = idata + IdJ;

                    DataType result(0);

                    // Process 4 items with each loop for efficiency.
                    while (Idata < lastgroup)
                    {
                        DataType const diff0 = Idata[0] - Jdata[0];
                        DataType const diff1 = Idata[1] - Jdata[1];
                        DataType const diff2 = Idata[2] - Jdata[2];
                        DataType const diff3 = Idata[3] - Jdata[3];
#if unrolledIncrement == 4
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
#endif
#if unrolledIncrement == 6
                        DataType const diff4 = Idata[4] - Jdata[4];
                        DataType const diff5 = Idata[5] - Jdata[5];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5;
#endif
#if unrolledIncrement == 8
                        DataType const diff4 = Idata[4] - Jdata[4];
                        DataType const diff5 = Idata[5] - Jdata[5];
                        DataType const diff6 = Idata[6] - Jdata[6];
                        DataType const diff7 = Idata[7] - Jdata[7];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
#endif
#if unrolledIncrement == 10
                        DataType const diff4 = Idata[4] - Jdata[4];
                        DataType const diff5 = Idata[5] - Jdata[5];
                        DataType const diff6 = Idata[6] - Jdata[6];
                        DataType const diff7 = Idata[7] - Jdata[7];
                        DataType const diff8 = Idata[8] - Jdata[8];
                        DataType const diff9 = Idata[9] - Jdata[9];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9;
#endif
#if unrolledIncrement == 12
                        DataType const diff4 = Idata[4] - Jdata[4];
                        DataType const diff5 = Idata[5] - Jdata[5];
                        DataType const diff6 = Idata[6] - Jdata[6];
                        DataType const diff7 = Idata[7] - Jdata[7];
                        DataType const diff8 = Idata[8] - Jdata[8];
                        DataType const diff9 = Idata[9] - Jdata[9];
                        DataType const diff10 = Idata[10] - Jdata[10];
                        DataType const diff11 = Idata[11] - Jdata[11];
                        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9 + diff10 * diff10 + diff11 * diff11;
#endif
                        Idata += unrolledIncrement;
                        Jdata += unrolledIncrement;
                    }
                    // Process last 0-3 pixels.
                    while (Idata < last)
                    {
                        DataType const diff0 = *Idata++ - *Jdata++;
                        result += diff0 * diff0;
                    }

                    nnDist[j] = result;
                }
            }
#endif
        }
    }
}

template <typename DataType>
inline void kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::setCovariance(EMatrixX<DataType> const &Covariance)
{
    covariance = Covariance;
    // check to see if the input covariance matrix is positive definite, or not
    if (!isSelfAdjointMatrixPositiveDefinite<EMatrixX<DataType>>(covariance))
    {
        // If the input covariance matrix is not positive definite, we force it
        forceSelfAdjointMatrixPositiveDefinite<EMatrixX<DataType>>(covariance);
    }
    // Inverse the positive definite covariance matrix
    EMatrixX<DataType> covarianceInverse = covariance.inverse();
    // Compute the Cholesky decomposition \f$ (LL^T) \f$ and retrieve factor \f$ L \f$ in the decomposition.
    matrixL = covarianceInverse.llt().matrixL();
}

template <typename DataType>
inline void kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::setCovariance(std::vector<DataType> const &Covariance)
{
    covariance = EMapType<DataType>(const_cast<DataType *>(Covariance.data()), this->dataDimension, this->dataDimension);
    // check to see if the input covariance matrix is positive definite, or not
    if (!isSelfAdjointMatrixPositiveDefinite<EMatrixX<DataType>>(covariance))
    {
        // If the input covariance matrix is not positive definite, we force it
        forceSelfAdjointMatrixPositiveDefinite<EMatrixX<DataType>>(covariance);
    }
    // Inverse the positive definite covariance matrix
    EMatrixX<DataType> covarianceInverse = covariance.inverse();
    // Compute the Cholesky decomposition \f$ (LL^T) \f$ and retrieve factor \f$ L \f$ in the decomposition.
    matrixL = covarianceInverse.llt().matrixL();
}

template <typename DataType>
inline void kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::setCovariance(DataType const *Covariance)
{
    covariance = EMapType<DataType>(const_cast<DataType *>(Covariance), this->dataDimension, this->dataDimension);
    // check to see if the input covariance matrix is positive definite, or not
    if (!isSelfAdjointMatrixPositiveDefinite<EMatrixX<DataType>>(covariance))
    {
        // If the input covariance matrix is not positive definite, we force it
        forceSelfAdjointMatrixPositiveDefinite<EMatrixX<DataType>>(covariance);
    }
    // Inverse the positive definite covariance matrix
    EMatrixX<DataType> covarianceInverse = covariance.inverse();
    // Compute the Cholesky decomposition \f$ (LL^T) \f$ and retrieve factor \f$ L \f$ in the decomposition.
    matrixL = covarianceInverse.llt().matrixL();
}

template <typename DataType>
inline EMatrixX<DataType> const &kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::Covariance() const { return this->covariance; }

template <typename DataType>
inline EMatrixX<DataType> &kNearestNeighbor<DataType, NeighborDistance::MAHALANOBIS>::Covariance() { return this->covariance; }

} // namespace umuq

#endif // UMUQ_KNEARESTNEIGHBORS
