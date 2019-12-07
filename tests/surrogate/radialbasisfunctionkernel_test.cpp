#include "surrogate/radialbasisfunctionkernel.hpp"
#include "environment.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *
 * Test to check linearKernel
 * \sa umuq::linearKernel
 */
TEST(linearKernel_test, HandlesKernelConstruction)
{
    umuq::linearKernel l;
    EXPECT_EQ(l.order(), 1);
    EXPECT_DOUBLE_EQ(l.phiZero(), 0);
    EXPECT_DOUBLE_EQ(l.evaluate(0), 0);
    EXPECT_DOUBLE_EQ(l.evaluate(1), 1);
    umuq::EMatrixXd A = umuq::EMatrixXd::Random(10, 10);
    EXPECT_DOUBLE_EQ((l.evaluate(A) - A).norm(), 0);
    EXPECT_DOUBLE_EQ(l.deriv(1), 1);
    EXPECT_DOUBLE_EQ((l.deriv(A).array() - 1).matrix().norm(), 0);
}

/*!
 * \ingroup Test_Module
 *
 * Test to check cubicKernel
 * \sa umuq::cubicKernel
 */
TEST(cubicKernel_test, HandlesKernelConstruction)
{
    umuq::cubicKernel c;
    EXPECT_EQ(c.order(), 2);
    EXPECT_DOUBLE_EQ(c.phiZero(), 0);
    EXPECT_DOUBLE_EQ(c.evaluate(0), 0);
    EXPECT_DOUBLE_EQ(c.evaluate(1), 1);
    EXPECT_DOUBLE_EQ(c.evaluate(2), 8);
    umuq::EMatrixXd A = umuq::EMatrixXd::Random(10, 10);
    EXPECT_NEAR((c.evaluate(A).array() - A.array().pow(3)).matrix().norm(), 0, 1e-14);
    EXPECT_DOUBLE_EQ(c.deriv(1), 3);
    EXPECT_NEAR((c.deriv(A).array() - 3 * A.array().pow(2)).matrix().norm(), 0, 1e-14);
}

/*!
 * \ingroup Test_Module
 *
 * Test to check thinPlateKernel
 * \sa umuq::thinPlateKernel
 */
TEST(thinPlateKernel_test, HandlesKernelConstruction)
{
    umuq::thinPlateKernel t;
    EXPECT_EQ(t.order(), 2);
    EXPECT_DOUBLE_EQ(t.phiZero(), 0);
    EXPECT_DOUBLE_EQ(t.evaluate(0), 0);
    EXPECT_DOUBLE_EQ(t.evaluate(1), std::log(1 + umuq::machinePrecision<double>));
    EXPECT_DOUBLE_EQ(t.evaluate(2), 4 * std::log(2 + umuq::machinePrecision<double>));
    umuq::EMatrixXd A = umuq::EMatrixXd::Random(10, 10).cwiseAbs();
    EXPECT_NEAR((t.evaluate(A) - (A.array().pow(2) * (A.array() + umuq::machinePrecision<double>).log()).matrix()).norm(), 0, 1e-14);
    EXPECT_NEAR(t.deriv(1), 1, 1e-14);
    EXPECT_NEAR((t.deriv(A) - (A.array() * (1 + 2 * (A.array() + umuq::machinePrecision<double>).log())).matrix()).norm(), 0, 1e-14);
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
