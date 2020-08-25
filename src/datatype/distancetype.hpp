#ifndef UMUQ_DISTANCETYPE_H
#define UMUQ_DISTANCETYPE_H

namespace umuq
{

/*! \enum DistanceTypes
 * \ingroup Numerics_Module
 *
 * \brief Distance types which can be used in %UMUQ
 *
 */
enum class DistanceTypes
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

} // namespace umuq

#endif // UMUQ_DISTANCETYPE
