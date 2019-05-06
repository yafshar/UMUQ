#include "core/core.hpp"
#include "environment.hpp"
#include "surrogate/polynomialtail.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *  
 * Test to check linearPolynomialTail 
 */
TEST(linearPolynomialTail_test, HandlesConstruction)
{
    umuq::linearPolynomialTail l;
    EXPECT_EQ(l.degree(), 1);
    umuq::EVectorXd Point = umuq::EVectorXd::Random(10);
    umuq::EVectorXd TPoint(11);
    TPoint << 1, Point;
    EXPECT_DOUBLE_EQ((l.evaluate(Point) - TPoint).norm(), 0);

    umuq::EMatrixXd Points = umuq::EMatrixXd::Random(10, 10);
    umuq::EMatrixXd TPoints(11, 10);
    TPoints.block(1, 0, 10, 10) = Points;
    TPoints.block(0, 0, 1, 10) = umuq::ERowVectorXd::Ones(10);

    EXPECT_DOUBLE_EQ((l.evaluate(Points) - TPoints).norm(), 0);
}

/*!
 * \ingroup Test_Module
 *  
 * Test to check constantPolynomialTail 
 */
TEST(constantPolynomialTail_test, HandlesConstruction)
{
    umuq::constantPolynomialTail l;
    EXPECT_EQ(l.degree(), 0);
    umuq::EVectorXd Point = umuq::EVectorXd::Random(10);
    EXPECT_DOUBLE_EQ(l.evaluate(Point).size(), 1);
    EXPECT_DOUBLE_EQ(l.evaluate(Point)[0], 1);

    umuq::EMatrixXd Points = umuq::EMatrixXd::Random(10, 10);
    EXPECT_DOUBLE_EQ((l.evaluate(Points) - umuq::EMatrixXd::Ones(1, 10)).norm(), 0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}
