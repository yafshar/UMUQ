#ifndef UMUQ_KNEARESTNEIGHBORS_H
#define UMUQ_KNEARESTNEIGHBORS_H

#include "datatype/distancetype.hpp"
#include "knearestneighborbase.hpp"

namespace umuq
{

/*! \class kNearestNeighbor
 * \ingroup Numerics_Module
 * 
 * \brief Finding K nearest neighbors in high dimensional spaces
 * 
 * \tparam DataType      Data type
 * \tparam DistanceType  DistanceTypes type for computing the distances to the nearest neighbors.
 *                       (Default is EUCLIDEAN distance) 
 *                       \sa umuq::DistanceTypes
 */
template <typename DataType, umuq::DistanceTypes DistanceType = umuq::DistanceTypes::EUCLIDEAN>
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
class kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> : public kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> : public kNearestNeighborBase<DataType, flann::L1<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> : public kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> : public kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> : public kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> : public kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> : public kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> &&other);

  private:
    kNearestNeighbor() = delete;
};
template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> : public kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> &&other);

  private:
    kNearestNeighbor() = delete;
};
template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> : public kNearestNeighborBase<DataType, flann::Hamming<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> : public kNearestNeighborBase<DataType, flann::HammingLUT>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> &&other);

  private:
    kNearestNeighbor() = delete;
};

template <typename DataType>
class kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> : public kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>
{
  public:
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> &&other);
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> const &other);
    kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> &&other);

  private:
    kNearestNeighbor() = delete;
};

/*! \class kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>
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
class kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> : public kNearestNeighborBase<DataType, flann::L2<DataType>>
{
  public:
    /*!
     * \brief Construct a new kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> object
     *
     * \param ndataPoints  Number of data points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Construct a new kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> object
     *
     * \param ndataPoints  Number of data points
     * \param nqueryPoints Number of query points
     * \param nDim         Dimension of each point
     * \param kNeighbors   k nearest neighbors to find
     */
    kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors);

    /*!
     * \brief Move construct a new kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> object
     *
     * \param other kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> object
     */
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> &&other);

    /*!
     * \brief Copy construct a new kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> object
     *
     * \param other kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> object
     */
    kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> const &other);

    /*!
     * \brief Move assignment
     *
     * \param other kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> object
     * \returns kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>&
     */
    kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> &operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> &&other);

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

template <typename DataType, umuq::DistanceTypes DistanceType>
kNearestNeighbor<DataType, DistanceType>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType, umuq::DistanceTypes DistanceType>
kNearestNeighbor<DataType, DistanceType>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType, umuq::DistanceTypes DistanceType>
kNearestNeighbor<DataType, DistanceType>::kNearestNeighbor(kNearestNeighbor<DataType, DistanceType> &&other) : kNearestNeighborBase<DataType, flann::L2<DataType>>(std::move(other))
{
}

template <typename DataType, umuq::DistanceTypes DistanceType>
kNearestNeighbor<DataType, DistanceType>::kNearestNeighbor(kNearestNeighbor<DataType, DistanceType> const &other) : kNearestNeighborBase<DataType, flann::L2<DataType>>(other)
{
}

template <typename DataType, umuq::DistanceTypes DistanceType>
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
kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> &&other) : kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> const &other) : kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> &kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::L2_SIMPLE> &>(kNearestNeighborBase<DataType, flann::L2_Simple<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L1<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L1<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> &&other) : kNearestNeighborBase<DataType, flann::L1<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> const &other) : kNearestNeighborBase<DataType, flann::L1<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> &kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::MANHATTAN> &>(kNearestNeighborBase<DataType, flann::L1<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> &&other) : kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> const &other) : kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> &kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::MINKOWSKI> &>(kNearestNeighborBase<DataType, flann::MinkowskiDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MAX>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MAX>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MAX>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> &&other) : kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MAX>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> const &other) : kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> &kNearestNeighbor<DataType, umuq::DistanceTypes::MAX>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::MAX> &>(kNearestNeighborBase<DataType, flann::MaxDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> &&other) : kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> const &other) : kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> &kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::HIST_INTERSECT> &>(kNearestNeighborBase<DataType, flann::HistIntersectionDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> &&other) : kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> const &other) : kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> &kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::HELLINGER> &>(kNearestNeighborBase<DataType, flann::HellingerDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> &&other) : kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> const &other) : kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> &kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::CHI_SQUARE> &>(kNearestNeighborBase<DataType, flann::ChiSquareDistance<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> &&other) : kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> const &other) : kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> &kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::KULLBACK_LEIBLER> &>(kNearestNeighborBase<DataType, flann::KL_Divergence<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::Hamming<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::Hamming<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> &&other) : kNearestNeighborBase<DataType, flann::Hamming<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> const &other) : kNearestNeighborBase<DataType, flann::Hamming<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> &kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING> &>(kNearestNeighborBase<DataType, flann::Hamming<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HammingLUT>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HammingLUT>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> &&other) : kNearestNeighborBase<DataType, flann::HammingLUT>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> const &other) : kNearestNeighborBase<DataType, flann::HammingLUT>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> &kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_LUT> &>(kNearestNeighborBase<DataType, flann::HammingLUT>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>(ndataPoints, nDim, kNeighbors)
{
}
template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors) {}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> &&other) : kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>(std::move(other))
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> const &other) : kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>(other)
{
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> &kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::HAMMING_POPCNT> &>(kNearestNeighborBase<DataType, flann::HammingPopcnt<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::kNearestNeighbor(int const ndataPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2<DataType>>(ndataPoints, nDim, kNeighbors),
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
kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::kNearestNeighbor(int const ndataPoints, int const nqueryPoints, int const nDim, int const kNeighbors) : kNearestNeighborBase<DataType, flann::L2<DataType>>(ndataPoints, nqueryPoints, nDim, kNeighbors),
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
kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> &&other) : kNearestNeighborBase<DataType, flann::L2<DataType>>(std::move(other))
{
    this->covariance = std::move(other.covariance);
    this->matrixL = std::move(other.matrixL);
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::kNearestNeighbor(kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> const &other) : kNearestNeighborBase<DataType, flann::L2<DataType>>(other)
{
    this->covariance = other.covariance;
    this->matrixL = other.matrixL;
}

template <typename DataType>
kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> &kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::operator=(kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> &&other)
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

    return static_cast<kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS> &>(kNearestNeighborBase<DataType, flann::L2<DataType>>::operator=(std::move(other)));
}

template <typename DataType>
void kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::buildIndex(DataType *idata)
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
void kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::buildIndex(DataType *idata, DataType *qdata)
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
inline void kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::setCovariance(EMatrixX<DataType> const &Covariance)
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
inline void kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::setCovariance(std::vector<DataType> const &Covariance)
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
inline void kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::setCovariance(DataType const *Covariance)
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
inline EMatrixX<DataType> const &kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::Covariance() const { return this->covariance; }

template <typename DataType>
inline EMatrixX<DataType> &kNearestNeighbor<DataType, umuq::DistanceTypes::MAHALANOBIS>::Covariance() { return this->covariance; }

} // namespace umuq

#endif // UMUQ_KNEARESTNEIGHBORS
