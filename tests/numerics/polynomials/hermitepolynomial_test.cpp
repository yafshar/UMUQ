#include "numerics/polynomials.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 */
TEST(HermiteBasis_test, HandlesInput)
{
    // Create an instance of a Hermite polynomial object
    umuq::HermitePolynomial<double> HPolynomial(1);

    EXPECT_DOUBLE_EQ(HPolynomial.hermite(0, 0.75), 1.);
    EXPECT_DOUBLE_EQ(HPolynomial.hermite(1, 0.75), 1.5);
    EXPECT_DOUBLE_EQ(HPolynomial.hermite(10, 0.75), 38740.4384765625);
    EXPECT_DOUBLE_EQ(HPolynomial.hermite(25, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(HPolynomial.hermite(25, 0.75), -9.7029819451106077507781088352e15);
    EXPECT_DOUBLE_EQ(HPolynomial.hermite(28, 0.75), 3.7538457078067672096408339776e18);

    double *PolynomialArray = nullptr;

    PolynomialArray = HPolynomial.hermite_array(100, 0.75);

    EXPECT_DOUBLE_EQ(PolynomialArray[0], 1.);
    EXPECT_DOUBLE_EQ(PolynomialArray[1], 1.5);
    EXPECT_DOUBLE_EQ(PolynomialArray[10], 38740.4384765625);
    EXPECT_DOUBLE_EQ(PolynomialArray[100], -1.4611185395125104593177790757e93);

    delete[] PolynomialArray;

    // Create an instance of a Hermite polynomial object after move assignment
    auto pp = std::move(HPolynomial);
    
    EXPECT_DOUBLE_EQ(pp.hermite(0, 0.75), 1.);
    EXPECT_DOUBLE_EQ(pp.hermite(1, 0.75), 1.5);
    EXPECT_DOUBLE_EQ(pp.hermite(10, 0.75), 38740.4384765625);
    EXPECT_DOUBLE_EQ(pp.hermite(25, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(pp.hermite(25, 0.75), -9.7029819451106077507781088352e15);
    EXPECT_DOUBLE_EQ(pp.hermite(28, 0.75), 3.7538457078067672096408339776e18);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
