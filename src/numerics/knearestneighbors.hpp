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
 * \tparam T             Data type
 * \tparam DistanceType  NeighborDistance type for computing the distances to the nearest neighbors.
 *                       (Default is EUCLIDEAN distance) 
 *                       \sa umuq::NeighborDistance
 */
template <typename T, NeighborDistance DistanceType = NeighborDistance::EUCLIDEAN>
class kNearestNeighbor : public kNearestNeighborBase<T, flann::L2<T>>
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
    kNearestNeighbor(kNearestNeighbor<T, DistanceType> &&other);

    /*!
     * \brief Copy construct a new kNearestNeighbor object
     *
     * \param other kNearestNeighbor object
     */
    kNearestNeighbor(kNearestNeighbor<T, DistanceType> const &other);

    /*!
     * \brief Move assignment
     *
     * \param other kNearestNeighbor object
     * \returns kNearestNeighbor<T>&
     */
    kNearestNeighbor<T, DistanceType> &operator=(kNearestNeighbor<T, DistanceType> &&other);

  private:
    /*!
     * \brief Explicitly prevent the default construct a new k Nearest Neighbor object
     * 
     */
    kNearestNeighbor() = delete;
};

template <typename T>
class kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> : public kNearestNeighborBase<T, flann::L2_Simple<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> const &other);
    kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> &operator=(kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename T>
class kNearestNeighbor<T, NeighborDistance::MANHATTAN> : public kNearestNeighborBase<T, flann::L1<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MANHATTAN> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MANHATTAN> const &other);
    kNearestNeighbor<T, NeighborDistance::MANHATTAN> &operator=(kNearestNeighbor<T, NeighborDistance::MANHATTAN> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename T>
class kNearestNeighbor<T, NeighborDistance::MINKOWSKI> : public kNearestNeighborBase<T, flann::MinkowskiDistance<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MINKOWSKI> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MINKOWSKI> const &other);
    kNearestNeighbor<T, NeighborDistance::MINKOWSKI> &operator=(kNearestNeighbor<T, NeighborDistance::MINKOWSKI> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename T>
class kNearestNeighbor<T, NeighborDistance::MAX> : public kNearestNeighborBase<T, flann::MaxDistance<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MAX> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MAX> const &other);
    kNearestNeighbor<T, NeighborDistance::MAX> &operator=(kNearestNeighbor<T, NeighborDistance::MAX> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename T>
class kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> : public kNearestNeighborBase<T, flann::HistIntersectionDistance<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> const &other);
    kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> &operator=(kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename T>
class kNearestNeighbor<T, NeighborDistance::HELLINGER> : public kNearestNeighborBase<T, flann::HellingerDistance<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HELLINGER> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HELLINGER> const &other);
    kNearestNeighbor<T, NeighborDistance::HELLINGER> &operator=(kNearestNeighbor<T, NeighborDistance::HELLINGER> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename T>
class kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> : public kNearestNeighborBase<T, flann::ChiSquareDistance<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> const &other);
    kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> &operator=(kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> &&other);

  private:
    kNearestNeighbor() = delete;
};
template <typename T>
class kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> : public kNearestNeighborBase<T, flann::KL_Divergence<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> const &other);
    kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> &operator=(kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> &&other);

  private:
    kNearestNeighbor() = delete;
};
template <typename T>
class kNearestNeighbor<T, NeighborDistance::HAMMING> : public kNearestNeighborBase<T, flann::Hamming<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING> const &other);
    kNearestNeighbor<T, NeighborDistance::HAMMING> &operator=(kNearestNeighbor<T, NeighborDistance::HAMMING> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename T>
class kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> : public kNearestNeighborBase<T, flann::HammingLUT>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> const &other);
    kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> &operator=(kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename T>
class kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> : public kNearestNeighborBase<T, flann::HammingPopcnt<T>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> &&other);
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> const &other);
    kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> &operator=(kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> &&other);

  private:
    kNearestNeighbor() = delete;
};

/*! \class kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>
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
class kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> : public kNearestNeighborBase<T, flann::L2<T>>
{
  public:
    /*!
     * \brief Construct a new kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> object
     *
     * \param ndataPoints  Number of data points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Construct a new kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> object
     *
     * \param ndataPoints  Number of data points
     * \param nqueryPoints Number of query points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Move construct a new kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> object
     *
     * \param other kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> object
     */
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> &&other);

    /*!
     * \brief Copy construct a new kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> object
     *
     * \param other kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> object
     */
    kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> const &other);

    /*!
     * \brief Move assignment
     *
     * \param other kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> object
     * \returns kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>&
     */
    kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> &operator=(kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> &&other);

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
    inline void setCovariance(EMatrixX<T> const &Covariance);

    /*!
     * \brief Set the Covariance object
     *
     * \param Covariance  The covariance matrix
     */
    inline void setCovariance(T const *Covariance);

    /*!
     * \brief Set the Covariance object
     *
     * \param Covariance  The covariance matrix
     */
    inline void setCovariance(std::vector<T> const &Covariance);

    /*!
     * \brief Access the covariance matrix.
     *
     * \returns Constant reference to the covariance matrix.
     */
    inline EMatrixX<T> const &Covariance() const;

    /*!
     * \brief Modify the covariance matrix.
     *
     * \returns Reference to the covariance matrix.
     */
    inline EMatrixX<T> &Covariance();

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
     *
     */
    EMatrixX<T> covariance;

    //! A view of the lower triangular matrix L from the Cholesky decomposition of inverse covariance matrix
    EMatrixX<T> matrixL;
};

template <typename T, NeighborDistance DistanceType>
kNearestNeighbor<T, DistanceType>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::L2<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T, NeighborDistance DistanceType>
kNearestNeighbor<T, DistanceType>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::L2<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T, NeighborDistance DistanceType>
kNearestNeighbor<T, DistanceType>::kNearestNeighbor(kNearestNeighbor<T, DistanceType> &&other) : kNearestNeighborBase<T, flann::L2<T>>(std::move(other))
{
}

template <typename T, NeighborDistance DistanceType>
kNearestNeighbor<T, DistanceType>::kNearestNeighbor(kNearestNeighbor<T, DistanceType> const &other) : kNearestNeighborBase<T, flann::L2<T>>(other)
{
}

template <typename T, NeighborDistance DistanceType>
kNearestNeighbor<T, DistanceType> &kNearestNeighbor<T, DistanceType>::operator=(kNearestNeighbor<T, DistanceType> &&other)
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

    return static_cast<kNearestNeighbor<T, DistanceType> &>(kNearestNeighborBase<T, flann::L2<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::L2_SIMPLE>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::L2_Simple<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::L2_SIMPLE>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::L2_Simple<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::L2_SIMPLE>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> &&other) : kNearestNeighborBase<T, flann::L2_Simple<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::L2_SIMPLE>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> const &other) : kNearestNeighborBase<T, flann::L2_Simple<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> &kNearestNeighbor<T, NeighborDistance::L2_SIMPLE>::operator=(kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::L2_SIMPLE> &>(kNearestNeighborBase<T, flann::L2_Simple<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MANHATTAN>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::L1<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::MANHATTAN>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::L1<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MANHATTAN>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MANHATTAN> &&other) : kNearestNeighborBase<T, flann::L1<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MANHATTAN>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MANHATTAN> const &other) : kNearestNeighborBase<T, flann::L1<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MANHATTAN> &kNearestNeighbor<T, NeighborDistance::MANHATTAN>::operator=(kNearestNeighbor<T, NeighborDistance::MANHATTAN> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::MANHATTAN> &>(kNearestNeighborBase<T, flann::L1<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MINKOWSKI>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::MinkowskiDistance<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::MINKOWSKI>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::MinkowskiDistance<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MINKOWSKI>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MINKOWSKI> &&other) : kNearestNeighborBase<T, flann::MinkowskiDistance<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MINKOWSKI>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MINKOWSKI> const &other) : kNearestNeighborBase<T, flann::MinkowskiDistance<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MINKOWSKI> &kNearestNeighbor<T, NeighborDistance::MINKOWSKI>::operator=(kNearestNeighbor<T, NeighborDistance::MINKOWSKI> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::MINKOWSKI> &>(kNearestNeighborBase<T, flann::MinkowskiDistance<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MAX>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::MaxDistance<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::MAX>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::MaxDistance<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MAX>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MAX> &&other) : kNearestNeighborBase<T, flann::MaxDistance<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MAX>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MAX> const &other) : kNearestNeighborBase<T, flann::MaxDistance<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MAX> &kNearestNeighbor<T, NeighborDistance::MAX>::operator=(kNearestNeighbor<T, NeighborDistance::MAX> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::MAX> &>(kNearestNeighborBase<T, flann::MaxDistance<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::HistIntersectionDistance<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::HistIntersectionDistance<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> &&other) : kNearestNeighborBase<T, flann::HistIntersectionDistance<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> const &other) : kNearestNeighborBase<T, flann::HistIntersectionDistance<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> &kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT>::operator=(kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::HIST_INTERSECT> &>(kNearestNeighborBase<T, flann::HistIntersectionDistance<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HELLINGER>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::HellingerDistance<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::HELLINGER>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::HellingerDistance<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HELLINGER>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HELLINGER> &&other) : kNearestNeighborBase<T, flann::HellingerDistance<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HELLINGER>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HELLINGER> const &other) : kNearestNeighborBase<T, flann::HellingerDistance<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HELLINGER> &kNearestNeighbor<T, NeighborDistance::HELLINGER>::operator=(kNearestNeighbor<T, NeighborDistance::HELLINGER> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::HELLINGER> &>(kNearestNeighborBase<T, flann::HellingerDistance<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::CHI_SQUARE>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::ChiSquareDistance<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::CHI_SQUARE>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::ChiSquareDistance<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::CHI_SQUARE>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> &&other) : kNearestNeighborBase<T, flann::ChiSquareDistance<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::CHI_SQUARE>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> const &other) : kNearestNeighborBase<T, flann::ChiSquareDistance<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> &kNearestNeighbor<T, NeighborDistance::CHI_SQUARE>::operator=(kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::CHI_SQUARE> &>(kNearestNeighborBase<T, flann::ChiSquareDistance<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::KL_Divergence<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::KL_Divergence<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> &&other) : kNearestNeighborBase<T, flann::KL_Divergence<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> const &other) : kNearestNeighborBase<T, flann::KL_Divergence<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> &kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER>::operator=(kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::KULLBACK_LEIBLER> &>(kNearestNeighborBase<T, flann::KL_Divergence<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::Hamming<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::Hamming<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING> &&other) : kNearestNeighborBase<T, flann::Hamming<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING> const &other) : kNearestNeighborBase<T, flann::Hamming<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING> &kNearestNeighbor<T, NeighborDistance::HAMMING>::operator=(kNearestNeighbor<T, NeighborDistance::HAMMING> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::HAMMING> &>(kNearestNeighborBase<T, flann::Hamming<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_LUT>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::HammingLUT>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_LUT>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::HammingLUT>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_LUT>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> &&other) : kNearestNeighborBase<T, flann::HammingLUT>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_LUT>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> const &other) : kNearestNeighborBase<T, flann::HammingLUT>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> &kNearestNeighbor<T, NeighborDistance::HAMMING_LUT>::operator=(kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::HAMMING_LUT> &>(kNearestNeighborBase<T, flann::HammingLUT>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::HammingPopcnt<T>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::HammingPopcnt<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> &&other) : kNearestNeighborBase<T, flann::HammingPopcnt<T>>(std::move(other))
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> const &other) : kNearestNeighborBase<T, flann::HammingPopcnt<T>>(other)
{
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> &kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT>::operator=(kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::HAMMING_POPCNT> &>(kNearestNeighborBase<T, flann::HammingPopcnt<T>>::operator=(std::move(other)));
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::L2<T>>(ndataPoints, nDim, kNeighbors),
                                                                                                                                    covariance(EMatrixX<T>::Identity(nDim, nDim)),
                                                                                                                                    matrixL(EMatrixX<T>::Identity(nDim, nDim))
{
    if (!(unrolledIncrement == 0 || unrolledIncrement == 4 || unrolledIncrement == 6 || unrolledIncrement == 8 || unrolledIncrement == 10 || unrolledIncrement == 12))
    {
        UMUQFAIL("The unrolled increment value of unrolledIncrement is not correctly set!");
    }
    this->withCovariance = true;
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<T, flann::L2<T>>(ndataPoints, nqueryPoints, nDim, kNeighbors),
                                                                                                                                                            covariance(EMatrixX<T>::Identity(nDim, nDim)),
                                                                                                                                                            matrixL(EMatrixX<T>::Identity(nDim, nDim))
{
    if (!(unrolledIncrement == 0 || unrolledIncrement == 4 || unrolledIncrement == 6 || unrolledIncrement == 8 || unrolledIncrement == 10 || unrolledIncrement == 12))
    {
        UMUQFAIL("The unrolled increment value of unrolledIncrement is not correctly set!");
    }
    this->withCovariance = true;
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> &&other) : kNearestNeighborBase<T, flann::L2<T>>(std::move(other))
{
    this->covariance = std::move(other.covariance);
    this->matrixL = std::move(other.matrixL);
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::kNearestNeighbor(kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> const &other) : kNearestNeighborBase<T, flann::L2<T>>(other)
{
    this->covariance = other.covariance;
    this->matrixL = other.matrixL;
}

template <typename T>
kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> &kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::operator=(kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> &&other)
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

    return static_cast<kNearestNeighbor<T, NeighborDistance::MAHALANOBIS> &>(kNearestNeighborBase<T, flann::L2<T>>::operator=(std::move(other)));
}

template <typename T>
void kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::buildIndex(T *idata)
{
    if (covariance.isIdentity(1e-3))
    {
        kNearestNeighborBase<T, flann::L2<T>>::buildIndex(idata);
    }
    else
    {
        // Copy the input data into a temporary array
        std::vector<T> inputData{idata, idata + this->nDataPoints * this->dataDimension};

        // Map the temporary array in Eigen format without memory copy
        EMapType<T> inputDataEMap(inputData.data(), this->nDataPoints, this->dataDimension);

        // Compute the input data matrix multiply by lower triangular matrix L from
        // the Cholesky decomposition of inverse covariance matrix
        inputDataEMap *= matrixL;

        // Map the data in flann matrix format
        flann::Matrix<T> dataset(inputData.data(), this->nDataPoints, this->dataDimension);

        // Construct an randomized kd-tree index using 4 kd-trees
        // For the number of parallel kd-trees to use (Good values are in the range [1..16])
        flann::Index<flann::L2<T>> index(dataset, flann::KDTreeIndexParams(4));
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
            T *nnDist = this->NearestNeighborsDistances(i);

#if unrolledIncrement == 0
            {
                T *last = idata + IdI + this->dataDimension;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->dataDimension;

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
                T *last = idata + IdI + this->dataDimension;
                T *lastgroup = last - unrolledIncrement + 1;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->dataDimension;

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
void kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::buildIndex(T *idata, T *qdata)
{
    if (covariance.isIdentity(1e-3))
    {
        kNearestNeighborBase<T, flann::L2<T>>::buildIndex(idata, qdata);
    }
    else
    {
        // Copy the input data into a temporary array
        std::vector<T> inputData{idata, idata + this->nDataPoints * this->dataDimension};

        // Copy the query data into a temporary array
        std::vector<T> queryData{qdata, qdata + this->nQueryDataPoints * this->dataDimension};

        // Map the temporary array in Eigen format without memory copy
        EMapType<T> inputDataEMap(inputData.data(), this->nDataPoints, this->dataDimension);

        // Map the temporary array in Eigen format without memory copy
        EMapType<T> queryDataEMap(queryData.data(), this->nQueryDataPoints, this->dataDimension);

        // Compute the input data matrix multiply by lower triangular matrix L from
        // the Cholesky decomposition of inverse covariance matrix
        inputDataEMap *= matrixL;

        queryDataEMap *= matrixL;

        // Map the data in flann matrix format
        flann::Matrix<T> dataset(inputData.data(), this->nDataPoints, this->dataDimension);

        // Construct an randomized kd-tree index using 4 kd-trees
        // For the number of parallel kd-trees to use (Good values are in the range [1..16])
        flann::Index<flann::L2<T>> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        flann::Matrix<T> query(queryData.data(), this->nQueryDataPoints, this->dataDimension);

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
            T *nnDist = this->NearestNeighborsDistances(i);

#if unrolledIncrement == 0
            {
                T *last = qdata + IdI + this->dataDimension;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->dataDimension;

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
                T *last = qdata + IdI + this->dataDimension;
                T *lastgroup = last - unrolledIncrement + 1;

                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->dataDimension;

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
inline void kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::setCovariance(EMatrixX<T> const &Covariance)
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
inline void kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::setCovariance(std::vector<T> const &Covariance)
{
    covariance = EMapType<T>(const_cast<T *>(Covariance.data()), this->dataDimension, this->dataDimension);
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
inline void kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::setCovariance(T const *Covariance)
{
    covariance = EMapType<T>(const_cast<T *>(Covariance), this->dataDimension, this->dataDimension);
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
inline EMatrixX<T> const &kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::Covariance() const { return this->covariance; }

template <typename T>
inline EMatrixX<T> &kNearestNeighbor<T, NeighborDistance::MAHALANOBIS>::Covariance() { return this->covariance; }

} // namespace umuq

#endif // UMUQ_KNEARESTNEIGHBORS
