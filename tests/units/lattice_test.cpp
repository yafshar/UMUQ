#include "units/lattice.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *
 * \brief Tests lattice class for the default construction
 *
 */
TEST(lattice_test, HandlesConstruction)
{
    // Create an instance of the lattice class
    umuq::lattice l;

    EXPECT_TRUE(l.getLatticeType() == umuq::LatticeType::NONE);

    auto BasisVectorLength = l.getBasisVectorLength();
    EXPECT_TRUE((BasisVectorLength.array() == 1.0).all());

    auto BasisVectorAngle = l.getBasisVectorAngle();
    EXPECT_TRUE((BasisVectorAngle.array() == static_cast<double>(M_PI / 2.)).all());

    auto RealSpaceBasisVector = l.getRealSpaceBasisVector();
    EXPECT_DOUBLE_EQ(RealSpaceBasisVector[0], 1.0);
    EXPECT_TRUE((RealSpaceBasisVector.segment<3>(1).array() == 0.0).all());
    EXPECT_DOUBLE_EQ(RealSpaceBasisVector[4], 1.0);
    EXPECT_TRUE((RealSpaceBasisVector.segment<3>(5).array() == 0.0).all());
    EXPECT_DOUBLE_EQ(RealSpaceBasisVector[8], 1.0);

    auto ReciprocalSpaceBasisVector = l.getReciprocalSpaceBasisVector();
    EXPECT_DOUBLE_EQ(ReciprocalSpaceBasisVector[0], 1.0);
    EXPECT_TRUE((ReciprocalSpaceBasisVector.segment<3>(1).array() == 0.0).all());
    EXPECT_DOUBLE_EQ(ReciprocalSpaceBasisVector[4], 1.0);
    EXPECT_TRUE((ReciprocalSpaceBasisVector.segment<3>(5).array() == 0.0).all());
    EXPECT_DOUBLE_EQ(ReciprocalSpaceBasisVector[8], 1.0);

    auto Volume = l.getRealSpaceVolume();
    EXPECT_DOUBLE_EQ(Volume, 1.0);

    // Create a point
    std::vector<double> pointF(3, 0.5);
    auto pointC = l.fractionalToCartesian(pointF);
    for (auto i : pointC)
    {
        EXPECT_DOUBLE_EQ(i, 0.5);
    }

    // Create multiple points
    std::vector<double> pointsF(4 * 3, 0.5);
    auto pointsC = l.fractionalToCartesian(pointsF);
    for (auto i : pointsC)
    {
        EXPECT_DOUBLE_EQ(i, 0.5);
    }
}

/*!
 * \ingroup Test_Module
 *
 * \brief Tests the lattice class for construction with bounding vectors
 *
 */
TEST(lattice_test, HandlesConstructionWithBoundingVectors)
{
    std::vector<double> BoundingVectors(9, 0.0);
    BoundingVectors[0] = 10.0;
    BoundingVectors[4] = 10.0;
    BoundingVectors[8] = 10.0;

    // Create an instance of the lattice class
    umuq::lattice l(BoundingVectors);

    auto BasisVectorLength = l.getBasisVectorLength();
    EXPECT_TRUE((BasisVectorLength.array() == 10.0).all());

    auto BasisVectorAngle = l.getBasisVectorAngle();
    EXPECT_TRUE((BasisVectorAngle.array() == static_cast<double>(M_PI / 2.)).all());

    // Create multiple points
    std::vector<double> pointsF(4 * 3, 0.5);

    auto pointsC = l.fractionalToCartesian(pointsF);
    for (auto i : pointsC)
    {
        EXPECT_DOUBLE_EQ(i, 5.0);
    }

    auto pointsF2 = l.cartesianToFractional(pointsC);
    for (auto pointsF_It = pointsF.begin(), pointsF2_It = pointsF2.begin(); pointsF_It != pointsF.end(); ++pointsF_It, ++pointsF2_It)
    {
        EXPECT_DOUBLE_EQ(*pointsF_It, *pointsF2_It);
    }

    // Update the point position in fractional coordinates
    pointsF[0] = 0.0;
    pointsF[1] = 0.0;
    pointsF[2] = 0.1;

    umuq::EVectorMapTypeConst<double> pointF_1(pointsF.data(), 3);
    umuq::EVectorMapTypeConst<double> pointF_2(pointsF.data() + 3, 3);

    auto pointC_1 = l.fractionalToCartesian(pointF_1);
    auto pointC_2 = l.fractionalToCartesian(pointF_2);

    EXPECT_DOUBLE_EQ((pointC_1 - pointC_2).norm(), l.cartesianDistanceBetweenFractionalPoints(pointF_1, pointF_2));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
