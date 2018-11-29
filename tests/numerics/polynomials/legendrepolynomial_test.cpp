#include "core/core.hpp"
#include "numerics/polynomials.hpp"
#include "gtest/gtest.h"

/*! 
 * \ingroup Test_Module
 */
TEST(LegendreBasis_test, HandlesInput)
{
    // Create an instance of a Legendre polynomial object
    umuq::LegendrePolynomial<double> LPolynomial(1);

    EXPECT_DOUBLE_EQ(LPolynomial.legendre(0, 0.5), 1.);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(0, -0.5), 1.);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(1, 1.0e-8), 1.0e-08);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(1, 0.5), 0.5);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(1, -0.5), -0.5);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(2, 0.0), -0.5);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(2, 0.5), -0.125);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(2, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(3, -0.5), 0.4375);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(3, 0.5), -0.4375);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(3, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(10, -0.5), -0.18822860717773438);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(10, 1.0e-8), -0.24609374999999864648);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(10, 0.5), -0.18822860717773437500);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(10, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(99, -0.5), 0.08300778172138770477);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(99, 1.0e-8), -7.958923738716563193e-08);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(99, 0.5), -0.08300778172138770477);
    EXPECT_NEAR(LPolynomial.legendre(99, 0.999), -0.3317727359254778874, 1e-14);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(99, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(LPolynomial.legendre(1000, 1.0), 1.0);
    EXPECT_NEAR(LPolynomial.legendre(1000, -0.5), -0.019168251091650277878, 1e-15);

    double *PolynomialArray = nullptr;

    PolynomialArray = LPolynomial.legendre_array(100, 0.5);

    EXPECT_DOUBLE_EQ(PolynomialArray[0], 1.);
    EXPECT_DOUBLE_EQ(PolynomialArray[1], 0.5);
    EXPECT_DOUBLE_EQ(PolynomialArray[2], -0.125);
    EXPECT_DOUBLE_EQ(PolynomialArray[3], -0.4375);
    EXPECT_DOUBLE_EQ(PolynomialArray[10], -0.18822860717773437500);
    EXPECT_DOUBLE_EQ(PolynomialArray[99], -0.08300778172138770477);
    EXPECT_DOUBLE_EQ(PolynomialArray[100], -0.06051802596186118687);

    delete[] PolynomialArray;
}
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}