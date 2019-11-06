#ifndef UMUQ_LATTICE_H
#define UMUQ_LATTICE_H

#include "core/core.hpp"
#include "io/io.hpp"
#include "misc/parser.hpp"
#include "numerics/eigenlib.hpp"

#include <cmath>
#include <cstddef>

#include <vector>

namespace umuq
{

/*!
 * \enum LatticeType
 * \ingroup Species_Module
 *
 * \brief The LatticeType
 *
 * The LatticeType in UMUQ are the ones for 3d problems.
 * Exception is the custom style which can be used for both 2d and 3d problems.
 */
enum class LatticeType
{
    /*! A style none lattice */
    NONE,
    /*! Simple Cubic lattice */
    SC,
    /*! Body-Centred Cubic lattice */
    BCC,
    /*! Face-Centred Cubic lattice */
    FCC,
    /*! Hexagonal Close-Packed lattice */
    HCP,
    /*! The diamond lattice */
    DIAMOND,
    /*! A style custom lattice */
    CUSTOM
};

/*! \class lattice
 * \ingroup Species_Module
 *
 * \brief General lattice class
 *
 * General lattice class with coordinate conversions.
 * A lattice is simply a set of points in space, determined by a unit cell,
 * that is replicated infinitely in all dimensions.
 */
class lattice
{
public:
    /*!
     * \brief Construct a new lattice object
     *
     * By default the new lattice object is cubic in nature and the vector lengths will be defined
     * as 1.0, and the angles to 90.0 degrees.
     */
    lattice();

    /*!
     * \brief Construct a new lattice object
     *
     * \param boundingVectors
     */
    explicit lattice(std::vector<double> const &boundingVectors);

    /*!
     * \brief Construct a new lattice object
     *
     * \param BasisVectorLength
     * \param BasisVectorAngle
     */
    lattice(EVector3d const &BasisVectorLength, EVector3d const &BasisVectorAngle);

    /*!
     * \brief Construct a new lattice object
     *
     * \param BasisVectorLength
     * \param BasisVectorAngle
     */
    lattice(std::vector<double> const &BasisVectorLength, std::vector<double> const &BasisVectorAngle);

    /*!
     * \brief Destroy the lattice object
     *
     */
    ~lattice() = default;

    /*!
     * \brief Move constructor, construct a new lattice object
     *
     * \param other lattice object
     */
    explicit lattice(lattice &&other);

    /*!
     * \brief Move assignment operator
     *
     * \param other lattice object
     *
     * \returns lattice& lattice object
     */
    lattice &operator=(lattice &&other);

    /*!
     * \brief Initialize the lattice components
     *
     */
    void init();

    /*!
     * \brief Get the Lattice Type object
     *
     * \returns LatticeType
     */
    inline LatticeType getLatticeType();

    /*!
     * \brief Set the Lattice Type object
     *
     * \param inLatticeType
     */
    inline void setLatticeType(LatticeType const &inLatticeType);

    /*!
     * \brief Get the Basis Vector Length
     *
     * \returns EVector3d
     */
    inline EVector3d getBasisVectorLength();

    /*!
     * \brief Set the Basis Vector Length
     *
     * \param BasisVectorLength
     */
    inline void setBasisVectorLength(EVector3d const &BasisVectorLength);

    /*!
     * \brief Get the Basis Vector Angle
     *
     * \returns EVector3d
     */
    inline EVector3d getBasisVectorAngle();

    /*!
     * \brief Set the Basis Vector Angle
     *
     * \param BasisVectorAngle
     */
    inline void setBasisVectorAngle(EVector3d const &BasisVectorAngle);

    /*!
     * \brief Get the Real Space Basis Vector
     *
     * \returns EVectorXd
     */
    inline EVectorXd getRealSpaceBasisVector();

    /*!
     * \brief Get the Reciprocal Space Basis Vector
     *
     * \returns EVectorXd
     */
    inline EVectorXd getReciprocalSpaceBasisVector();

    /*!
     * \brief Get the Translation Vector
     *
     * \returns EVector3d
     */
    inline EVector3d getTranslationVector();

    /*!
     * \brief Get the Metric Tensor
     *
     * \returns EVector6d
     */
    inline EVector6d getMetricTensor();

    /*!
     * \brief Get the Real Space Volume
     *
     * \returns double Volume of the real space
     */
    inline double getRealSpaceVolume();

    /*!
     * \brief Get the Reciprocal Space Volume
     *
     * \returns double Volume of the reciprocal space
     */
    inline double getReciprocalSpaceVolume();

    /*!
     * \brief Converts the Cartesian coordinates of a point to fractional coordinates
     *
     * \param point  Cartesian coordinates of a point (3-dimensional)
     *
     * \returns EVector3d Fractional coordinates of the point (3-dimensional)
     */
    EVector3d cartesianToFractional(EVector3d const &point);

    /*!
     * \brief Converts the Cartesian coordinates of multiple points to fractional coordinates
     *
     * \param points  Cartesian coordinates of multiple points (in 3-dimensions)
     *
     * \returns std::vector<double> Fractional coordinates of points
     */
    std::vector<double> cartesianToFractional(std::vector<double> const &points);

    /*!
     * \brief Converts the fractional coordinates of a point to Cartesian coordinates
     *
     * \param point  Fractional coordinates of a point (3-dimensional)
     *
     * \returns EVector3d Cartesian coordinates of the point (3-dimensional)
     */
    EVector3d fractionalToCartesian(EVector3d const &point);

    /*!
     * \brief Converts the fractional coordinates of multiple points to Cartesian coordinates
     *
     * \param points   Fractional coordinates of multiple points (in 3-dimensions)
     *
     * \returns std::vector<double> Cartesian coordinates of the points
     */
    std::vector<double> fractionalToCartesian(std::vector<double> const &points);

    /*!
     * \brief Computes the Cartesian distance between two points with fractional coordinates
     *
     * \param point1  Fractional coordinates of a point (3-dimensional)
     * \param point2  Fractional coordinates of a point (3-dimensional)
     *
     * \returns double The Cartesian distance between two points
     */
    double cartesianDistanceBetweenFractionalPoints(EVector3d const &point1, EVector3d const &point2);

private:
    /*!
     * \brief Delete a lattice object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    lattice(lattice const &) = delete;

    /*!
     * \brief Delete a lattice object assignment
     *
     * Avoiding implicit copy assignment.
     */
    lattice &operator=(lattice const &) = delete;

public:
    /*! Lattice type */
    LatticeType latticeType;

    /*! Basis vectors lengths in Cartesian space */
    EVector3d basisVectorLength;

    /*! Basis vectors angles in Cartesian space */
    EVector3d basisVectorAngle;

    /*! Real space basis vectors */
    EVectorXd realSpaceBasisVector;

    /*! Reciprocal space basis vectors */
    EVectorXd reciprocalSpaceBasisVector;

    /*! Metric for the space */
    EVector6d metricTensor;

    /*! Cartesian space volume of lattice */
    double volume;
};

lattice::lattice() : latticeType(LatticeType::NONE),
                     basisVectorLength(EVector3d::Ones()),
                     basisVectorAngle(EVector3d::Ones() * static_cast<double>(M_PI / 2.0)),
                     realSpaceBasisVector(EVectorXd::Zero(9)),
                     reciprocalSpaceBasisVector(EVectorXd::Zero(9))
{
    init();
}

lattice::lattice(std::vector<double> const &boundingVectors) : latticeType(LatticeType::NONE)
{
    if (boundingVectors.size() != 9)
    {
        UMUQFAIL("The bounding vectors with wrong size ", boundingVectors.size(), " != 9 !");
    }

    EVectorXd BoundingVectors = EVectorMapTypeConst<double>(boundingVectors.data(), 9);

    // Compute the length of the bounding vectors
    basisVectorLength[0] = BoundingVectors.segment<3>(0).norm();
    basisVectorLength[1] = BoundingVectors.segment<3>(3).norm();
    basisVectorLength[2] = BoundingVectors.segment<3>(6).norm();
    if ((basisVectorLength.array() < UTHRESHOLD).any())
    {
        UMUQFAIL("The input vector length should be positive number bigger than UMUQ threshold!");
    }

    // Normalize the bounding vectors
    BoundingVectors.segment<3>(0) /= basisVectorLength[0];
    BoundingVectors.segment<3>(3) /= basisVectorLength[1];
    BoundingVectors.segment<3>(6) /= basisVectorLength[2];

    // Compute the angles between bounding vectors
    double dot = BoundingVectors.segment<3>(3).dot(BoundingVectors.segment<3>(6));
    dot = (dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot));
    basisVectorAngle[0] = std::acos(dot);

    dot = BoundingVectors.segment<3>(0).dot(BoundingVectors.segment<3>(6));
    dot = (dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot));
    basisVectorAngle[1] = std::acos(dot);

    dot = BoundingVectors.segment<3>(0).dot(BoundingVectors.segment<3>(3));
    dot = (dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot));
    basisVectorAngle[2] = std::acos(dot);

    realSpaceBasisVector = EVectorXd::Zero(9);
    reciprocalSpaceBasisVector = EVectorXd::Zero(9);

    init();
}

lattice::lattice(EVector3d const &BasisVectorLength, EVector3d const &BasisVectorAngle) : latticeType(LatticeType::NONE),
                                                                                          basisVectorLength(BasisVectorLength),
                                                                                          basisVectorAngle(BasisVectorAngle),
                                                                                          realSpaceBasisVector(EVectorXd::Zero(9)),
                                                                                          reciprocalSpaceBasisVector(EVectorXd::Zero(9))
{
    if ((basisVectorLength.array() < 0.0).any())
    {
        UMUQFAIL("The input vector length should be positive!");
    }
    if ((basisVectorAngle.array() < 0.0).any())
    {
        UMUQFAIL("The input vector angle should be positive!");
    }
    if ((basisVectorAngle.array() > static_cast<double>(M_PI)).any())
    {
        UMUQFAIL("The input vector angle should be smaller than 180!");
    }
    init();
}

lattice::lattice(std::vector<double> const &BasisVectorLength, std::vector<double> const &BasisVectorAngle) : latticeType(LatticeType::NONE),
                                                                                                              basisVectorLength(BasisVectorLength.data()),
                                                                                                              basisVectorAngle(BasisVectorAngle.data()),
                                                                                                              realSpaceBasisVector(EVectorXd::Zero(9)),
                                                                                                              reciprocalSpaceBasisVector(EVectorXd::Zero(9))
{
    if ((basisVectorLength.array() < 0.0).any())
    {
        UMUQFAIL("The input vector length should be positive!");
    }
    if ((basisVectorAngle.array() < 0.0).any())
    {
        UMUQFAIL("The input vector angle should be positive!");
    }
    if ((basisVectorAngle.array() > static_cast<double>(M_PI)).any())
    {
        UMUQFAIL("The input vector angle should be smaller than 180!");
    }
    init();
}

lattice::lattice(lattice &&other)
{
    latticeType = other.latticeType;
    basisVectorLength = std::move(other.basisVectorLength);
    basisVectorAngle = std::move(other.basisVectorAngle);
    realSpaceBasisVector = std::move(other.realSpaceBasisVector);
    reciprocalSpaceBasisVector = std::move(other.reciprocalSpaceBasisVector);
    metricTensor = std::move(other.metricTensor);
    volume = other.volume;
}

lattice &lattice::operator=(lattice &&other)
{
    latticeType = other.latticeType;
    basisVectorLength = std::move(other.basisVectorLength);
    basisVectorAngle = std::move(other.basisVectorAngle);
    realSpaceBasisVector = std::move(other.realSpaceBasisVector);
    reciprocalSpaceBasisVector = std::move(other.reciprocalSpaceBasisVector);
    metricTensor = std::move(other.metricTensor);
    volume = other.volume;

    return *this;
}

void lattice::init()
{
    // Compute the Cos and Sin of the basis vector angles
    EVector3d basisVectorAngleCos = basisVectorAngle.array().cos();
    EVector3d basisVectorAngleSin = basisVectorAngle.array().sin();

    // For the first axis project the length along the x-axis
    realSpaceBasisVector[0] = basisVectorLength[0];

    // For the second axis keep it in the xy-plane and rotate through angle between axes
    realSpaceBasisVector[3] = basisVectorLength[1] * basisVectorAngleCos[2];
    realSpaceBasisVector[4] = basisVectorLength[1] * basisVectorAngleSin[2];

    realSpaceBasisVector.segment<3>(3) = (UTHRESHOLD < realSpaceBasisVector.segment<3>(3).array().abs()).select(realSpaceBasisVector.segment<3>(3), 0.0);
    // The third axis needs to rotate relative to the two previously-defined vectors.

    // (1) Rotate away from the x-axis by angle basisVectorAngle[1]
    realSpaceBasisVector[6] = basisVectorLength[2] * basisVectorAngleCos[1];
    realSpaceBasisVector[8] = basisVectorLength[2] * basisVectorAngleSin[1];

    // (2) Do a constrained rotation around the x-axis such that the angle becomes basisVectorAngle[0]
    double const c = (basisVectorAngleCos[2] * basisVectorAngleCos[1] - basisVectorAngleCos[0]) / (basisVectorAngleSin[1] * basisVectorAngleSin[2]);

    realSpaceBasisVector[7] = -realSpaceBasisVector[8] * c;
    realSpaceBasisVector[8] *= std::sqrt(1 - c * c);

    realSpaceBasisVector.segment<3>(6) = (UTHRESHOLD < realSpaceBasisVector.segment<3>(6).array().abs()).select(realSpaceBasisVector.segment<3>(6), 0.0);

    // Construct the reciprocal space basis
    reciprocalSpaceBasisVector << realSpaceBasisVector.segment<3>(3).cross(realSpaceBasisVector.segment<3>(6)),
        realSpaceBasisVector.segment<3>(6).cross(realSpaceBasisVector.segment<3>(0)),
        realSpaceBasisVector.segment<3>(0).cross(realSpaceBasisVector.segment<3>(3));

    // Compute the Volume
    volume = realSpaceBasisVector.segment<3>(0).dot(reciprocalSpaceBasisVector.segment<3>(0));

    double const reciprocalVolume = 1.0 / volume;

    reciprocalSpaceBasisVector *= reciprocalVolume;

    reciprocalSpaceBasisVector = (UTHRESHOLD < reciprocalSpaceBasisVector.array().abs()).select(reciprocalSpaceBasisVector, 0.0);

    //  Create our metric:
    std::size_t k = 0;
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j <= i; j++)
        {
            metricTensor[k++] = realSpaceBasisVector.segment<3>(i * 3).dot(realSpaceBasisVector.segment<3>(j * 3));
        }
    }
}

inline LatticeType lattice::getLatticeType() { return latticeType; }

inline void lattice::setLatticeType(LatticeType const &inLatticeType) { latticeType = inLatticeType; }

inline EVector3d lattice::getBasisVectorLength() { return basisVectorLength; }

inline void lattice::setBasisVectorLength(EVector3d const &BasisVectorLength)
{
    if ((BasisVectorLength.array() < 0.0).any())
    {
        UMUQFAIL("The input vector length should be positive!");
    }
    basisVectorLength = BasisVectorLength;
    init();
}

//  lattice-edge angle accessors:
inline EVector3d lattice::getBasisVectorAngle() { return basisVectorAngle; }

inline void lattice::setBasisVectorAngle(EVector3d const &BasisVectorAngle)
{
    if ((basisVectorAngle.array() < 0.0).any())
    {
        UMUQFAIL("The input vector angle should be positive!");
    }
    if ((basisVectorAngle.array() > static_cast<double>(M_PI)).any())
    {
        UMUQFAIL("The input vector angle should be smaller than 180!");
    }
    basisVectorLength = BasisVectorAngle;
    init();
}

inline EVectorXd lattice::getRealSpaceBasisVector() { return realSpaceBasisVector; }

inline EVectorXd lattice::getReciprocalSpaceBasisVector() { return reciprocalSpaceBasisVector; }

// The translation vector for the lattice
inline EVector3d lattice::getTranslationVector()
{
    EVector3d TranslationVector;
    TranslationVector << realSpaceBasisVector[0] + realSpaceBasisVector[3] + realSpaceBasisVector[6],
        realSpaceBasisVector[1] + realSpaceBasisVector[4] + realSpaceBasisVector[7],
        realSpaceBasisVector[2] + realSpaceBasisVector[5] + realSpaceBasisVector[8];
    return TranslationVector;
}

inline EVector6d lattice::getMetricTensor() { return metricTensor; }

inline double lattice::getRealSpaceVolume() { return volume; };

inline double lattice::getReciprocalSpaceVolume() { return (1.0 / volume); };

EVector3d lattice::cartesianToFractional(EVector3d const &point)
{
    EVector3d Point;
    Point << point.dot(reciprocalSpaceBasisVector.segment<3>(0)),
        point.dot(reciprocalSpaceBasisVector.segment<3>(3)),
        point.dot(reciprocalSpaceBasisVector.segment<3>(6));
    Point = (UTHRESHOLD < Point.array().abs()).select(Point, 0.0);
    return Point;
}

std::vector<double> lattice::cartesianToFractional(std::vector<double> const &points)
{
    // Get the total size of the vector
    std::size_t const n = points.size();

#ifdef DEBUG
    if (n % 3 != 0)
    {
        UMUQFAIL("Vector of points has a wrong dimension mod(", n, " , 3) != 0 !");
    }
#endif
    std::vector<double> Points(n);

    // If only we have one points
    if (n == 3)
    {
        Points[0] = points[0] * reciprocalSpaceBasisVector[0] + points[1] * reciprocalSpaceBasisVector[1] + points[2] * reciprocalSpaceBasisVector[2];
        Points[1] = points[0] * reciprocalSpaceBasisVector[3] + points[1] * reciprocalSpaceBasisVector[4] + points[2] * reciprocalSpaceBasisVector[5];
        Points[2] = points[0] * reciprocalSpaceBasisVector[6] + points[1] * reciprocalSpaceBasisVector[7] + points[2] * reciprocalSpaceBasisVector[8];
    }
    else
    {
        EMapTypeConst<double> pointMatrix(points.data(), n / 3, 3);
        EMapType<double> PointMatrix(Points.data(), n / 3, 3);
        EMapTypeConst<double> reciprocalSpaceBasisMatrix(reciprocalSpaceBasisVector.data(), 3, 3);
        PointMatrix = pointMatrix * reciprocalSpaceBasisMatrix;
    }
    for (auto i : Points)
    {
        if (std::abs(i) <= UTHRESHOLD)
        {
            i = 0.0;
        }
    }
    return Points;
}

EVector3d lattice::fractionalToCartesian(EVector3d const &point)
{
    EVector3d Point = realSpaceBasisVector.segment<3>(0) * point[0];
    Point += realSpaceBasisVector.segment<3>(3) * point[1];
    Point += realSpaceBasisVector.segment<3>(6) * point[2];
    Point = (UTHRESHOLD < Point.array().abs()).select(Point, 0.0);
    return Point;
}

std::vector<double> lattice::fractionalToCartesian(std::vector<double> const &points)
{
    // Get the total size of the vector
    std::size_t const n = points.size();

#ifdef DEBUG
    if (n % 3 != 0)
    {
        UMUQFAIL("Vector of points has a wrong dimension mod(", n, " , 3) != 0 !");
    }
#endif

    std::vector<double> Points(n);

    // If only we have one points
    if (n == 3)
    {
        Points[0] = points[0] * realSpaceBasisVector[0] + points[1] * realSpaceBasisVector[3] + points[2] * realSpaceBasisVector[6];
        Points[1] = points[0] * realSpaceBasisVector[1] + points[1] * realSpaceBasisVector[4] + points[2] * realSpaceBasisVector[7];
        Points[2] = points[0] * realSpaceBasisVector[2] + points[1] * realSpaceBasisVector[5] + points[2] * realSpaceBasisVector[8];
    }
    else
    {
        EMapTypeConst<double> pointMatrix(points.data(), n / 3, 3);
        EMapType<double> PointMatrix(Points.data(), n / 3, 3);
        EMapTypeConst<double> realSpaceBasisMatrix(realSpaceBasisVector.data(), 3, 3);
        PointMatrix = pointMatrix * realSpaceBasisMatrix;
    }
    for (auto i : Points)
    {
        if (std::abs(i) <= UTHRESHOLD)
        {
            i = 0.0;
        }
    }
    return Points;
}

double lattice::cartesianDistanceBetweenFractionalPoints(EVector3d const &point1, EVector3d const &point2)
{
    EVector3d pointDiff = point1 - point2;
    double dist = (pointDiff[0] * metricTensor[0] + pointDiff[1] * metricTensor[1] + pointDiff[2] * metricTensor[3]) * pointDiff[0] +
                  (pointDiff[0] * metricTensor[1] + pointDiff[1] * metricTensor[2] + pointDiff[2] * metricTensor[4]) * pointDiff[1] +
                  (pointDiff[0] * metricTensor[3] + pointDiff[1] * metricTensor[4] + pointDiff[2] * metricTensor[5]) * pointDiff[2];
    return std::sqrt(dist);
}

} // namespace umuq

#endif // UMUQ_LATTICE_H
