#include "numerics/polynomials.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 */
TEST(ChebyshevBasis_test, HandlesInput)
{
    // Create an instance of a Chebyshev polynomial object
    umuq::ChebyshevPolynomial<double> CPolynomial(1);

    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(0, 0.75), 1.);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(1, 0.75), 0.75);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(1, -0.25), -0.25);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(4, 0.999), 0.984039968008000);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(4, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(10, 0.75), 0.586425781250000);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(10, -0.1), -0.538892748800000);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(10, 1.75), 5.3903809082031315e+04);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(25, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(25, 0.75), 0.710069283843040);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(25, -0.5), -0.5);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(28, 0.75), 0.18276030011475086);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(28, -0.1), -0.943782337026093);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(500, 1.0/3.0), 0.96311412681708752);

    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(0, 0.75, true), 1.);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(1, 0.75, true), 1.5);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(1, -0.25, true), -0.5);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(4, 0.999, true), 4.960083936016000);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(4, 1.0, true), 5.0);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(10, 0.75, true), 1.504882812500000);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(10, -0.1, true), -0.454230937600000);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(25, 0.0, true), 0.0);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(25, 0.75, true), -0.08834114670753479);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(25, -0.5, true), -1.0);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(28, 0.75, true), 1.297556120902300);
    EXPECT_DOUBLE_EQ(CPolynomial.chebyshev(28, -0.1, true), -0.977005625121864);
    EXPECT_NEAR(CPolynomial.chebyshev(500, 1.0/3.0, true), 0.86797529488884995, 1e-14);

    double *PolynomialArray = nullptr;

    PolynomialArray = CPolynomial.chebyshev_array(100, 0.75);

    EXPECT_DOUBLE_EQ(PolynomialArray[0], 1.);
    EXPECT_DOUBLE_EQ(PolynomialArray[1], 0.75);
    EXPECT_DOUBLE_EQ(PolynomialArray[10], 0.586425781250000);
    EXPECT_DOUBLE_EQ(PolynomialArray[100], -0.999858988315193);

    delete[] PolynomialArray;

    // Create an instance of a Chebyshev polynomial object after move assignment
    auto pp = std::move(CPolynomial);

    EXPECT_DOUBLE_EQ(pp.chebyshev(0, 0.75), 1.);
    EXPECT_DOUBLE_EQ(pp.chebyshev(1, 0.75), 0.75);
    EXPECT_DOUBLE_EQ(pp.chebyshev(1, -0.25), -0.25);
    EXPECT_DOUBLE_EQ(pp.chebyshev(4, 0.999), 0.984039968008000);
    EXPECT_DOUBLE_EQ(pp.chebyshev(4, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(pp.chebyshev(10, 0.75), 0.586425781250000);
    EXPECT_DOUBLE_EQ(pp.chebyshev(10, -0.1), -0.538892748800000);
    EXPECT_DOUBLE_EQ(pp.chebyshev(10, 1.75), 5.3903809082031315e+04);
    EXPECT_DOUBLE_EQ(pp.chebyshev(25, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(pp.chebyshev(25, 0.75), 0.710069283843040);
    EXPECT_DOUBLE_EQ(pp.chebyshev(25, -0.5), -0.5);
    EXPECT_DOUBLE_EQ(pp.chebyshev(28, 0.75), 0.18276030011475086);
    EXPECT_DOUBLE_EQ(pp.chebyshev(28, -0.1), -0.943782337026093);
    EXPECT_DOUBLE_EQ(pp.chebyshev(500, 1.0/3.0), 0.96311412681708752);    
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
