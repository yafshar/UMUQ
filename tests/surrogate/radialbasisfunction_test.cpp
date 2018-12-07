#include "core/core.hpp"
#include "environment.hpp"
#include "surrogate/radialbasisfunction.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *  
 * Test to check radialBasisFunction 
 * \sa umuq::radialBasisFunction
 */
TEST(radialBasisFunction_test, HandlesConstruction)
{
    int const nDimensions = 2;
    int const maxNumPoints = 1000;

    umuq::EMatrixXd Points = (umuq::EMatrixXd::Random(nDimensions, maxNumPoints) + umuq::EMatrixXd::Ones(nDimensions, maxNumPoints)) / 2;
    umuq::EVectorXd FunctionValues = (Points.row(1).array() * Points.row(0).array().sin() + Points.row(0).array() * Points.row(1).array().cos()).matrix().transpose();

    umuq::radialBasisFunction<umuq::cubicKernel, umuq::linearPolynomialTail> RBF(nDimensions, maxNumPoints, 0);
    EXPECT_TRUE(RBF.addPoint(Points, FunctionValues));
    EXPECT_TRUE(RBF.fit());

    EXPECT_TRUE(nDimensions == RBF.getNumDimensions());
    EXPECT_TRUE(maxNumPoints == RBF.getNumPoints());
    EXPECT_TRUE((Points - RBF.getCurrentPoints()).norm() <= umuq::machinePrecision<double>);
    EXPECT_TRUE((FunctionValues - RBF.getFunctionValues()).norm() <= umuq::machinePrecision<double>);

    for (auto i = 0; i < maxNumPoints; ++i)
    {
        EXPECT_TRUE((Points.col(i) - RBF.getCurrentPoints(i)).norm() <= umuq::machinePrecision<double>);
        EXPECT_TRUE(std::abs(FunctionValues(i) - RBF.getFunctionValues(i)) <= umuq::machinePrecision<double>);
    }
}

/*!
 * \ingroup Test_Module
 *  
 * Test to check radialBasisFunction 
 * \sa umuq::radialBasisFunction
 * 
 * The test function is \f$ f(x,y) = y\sin (x) + x\cos (y) \f$ 
 */
TEST(radialBasisFunction_test, HandlesFunctions)
{
    int const nDimensions = 2;
    int const maxNumPoints = 1000;
    int const nQueryPoints = 10;

    umuq::EMatrixXd Points = (umuq::EMatrixXd::Random(nDimensions, maxNumPoints) + umuq::EMatrixXd::Ones(nDimensions, maxNumPoints)) / 2;
    umuq::EVectorXd FunctionValues = (Points.row(1).array() * Points.row(0).array().sin() + Points.row(0).array() * Points.row(1).array().cos()).matrix().transpose();

    umuq::EMatrixXd QPoints = (umuq::EMatrixXd::Random(nDimensions, nQueryPoints) + umuq::EMatrixXd::Ones(nDimensions, nQueryPoints)) / 2;
    umuq::EVectorXd QFunctionValues = (QPoints.row(1).array() * QPoints.row(0).array().sin() + QPoints.row(0).array() * QPoints.row(1).array().cos()).matrix().transpose();

    umuq::radialBasisFunction<umuq::cubicKernel, umuq::linearPolynomialTail> RBF(nDimensions, maxNumPoints, 0);

    EXPECT_TRUE(RBF.addPoint(Points, FunctionValues));

    EXPECT_TRUE(RBF.fit());

    // Evaluate at the center to see that we are interpolating
    {
        umuq::EVectorXd PointsEvaluation = RBF.evaluate(Points);
        EXPECT_FALSE((PointsEvaluation - FunctionValues).cwiseAbs().maxCoeff() > 1e-10);
    }

    // Evaluate at some other points
    {
        umuq::EVectorXd PointsEvaluation = RBF.evaluate(QPoints);
        EXPECT_FALSE((PointsEvaluation - QFunctionValues).cwiseAbs().maxCoeff() > 1e-3);

        umuq::EMatrixXd Distance = umuq::L2Distance<double>(QPoints, Points);
        EXPECT_FALSE((PointsEvaluation - RBF.evaluate(QPoints, Distance)).norm() > 1e-10);
    }

    // Look at derivatives
    {
        umuq::EMatrixXd DerivativeValues(nDimensions, nQueryPoints);
        DerivativeValues.row(0) = (QPoints.row(1).array() * QPoints.row(0).array().cos() + QPoints.row(1).array().cos()).matrix();
        DerivativeValues.row(1) = (QPoints.row(0).array().sin() - QPoints.row(0).array() * QPoints.row(1).array().sin()).matrix();

        for (auto i = 0; i < nQueryPoints; ++i)
        {
            umuq::EVectorXd Derivative = RBF.deriv(QPoints.col(i));
            EXPECT_FALSE((DerivativeValues.col(i) - Derivative).norm() >= 1e-2);
        }
    }

    // Reset the Object
    RBF.reset();

    // Add all points again
    EXPECT_TRUE(RBF.addPoint(Points.block(0, 0, nDimensions, maxNumPoints - 10), FunctionValues.head(maxNumPoints - 10)));
    EXPECT_TRUE(RBF.addPoint(Points.block(0, maxNumPoints - 10, nDimensions, 5), FunctionValues.segment(maxNumPoints - 10, 5)));
    for (int i = maxNumPoints - 5; i < maxNumPoints; i++)
    {
        EXPECT_TRUE(RBF.addPoint(Points.col(i), FunctionValues(i)));
    }
    EXPECT_FALSE(RBF.addPoint(Points.col(0), FunctionValues(0)));
    EXPECT_TRUE(RBF.fit());
}

/*!
 * \ingroup Test_Module
 *  
 * Test to check radialBasisFunctionCap 
 * \sa umuq::radialBasisFunctionCap
 */
TEST(radialBasisFunctionCap_test, HandlesCapFunctions)
{
    int const nDimensions = 2;
    int const maxNumPoints = 1000;
    int const nQueryPoints = 10;

    umuq::EMatrixXd Points = (umuq::EMatrixXd::Random(nDimensions, maxNumPoints) + umuq::EMatrixXd::Ones(nDimensions, maxNumPoints)) / 2;
    umuq::EVectorXd FunctionValues = (Points.row(1).array() * Points.row(0).array().sin() + Points.row(0).array() * Points.row(1).array().cos()).matrix().transpose();

    // Set half the points to crazy values
    FunctionValues.segment(0, maxNumPoints / 2 - 2).fill(std::numeric_limits<double>::max());

    umuq::EMatrixXd QPoints = (umuq::EMatrixXd::Random(nDimensions, nQueryPoints) + umuq::EMatrixXd::Ones(nDimensions, nQueryPoints)) / 2;
    umuq::EVectorXd QFunctionValues = (QPoints.row(1).array() * QPoints.row(0).array().sin() + QPoints.row(0).array() * QPoints.row(1).array().cos()).matrix().transpose();

    umuq::radialBasisFunctionCap<umuq::cubicKernel, umuq::linearPolynomialTail> CRBF(nDimensions, maxNumPoints, 0);

    EXPECT_TRUE(CRBF.addPoint(Points, FunctionValues));
    EXPECT_TRUE(CRBF.fit());

    // Evaluate at some other points
    {
        umuq::EVectorXd PointsEvaluation = CRBF.evaluate(QPoints);
        EXPECT_FALSE((PointsEvaluation - QFunctionValues).cwiseAbs().maxCoeff() > 2);
    }
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
