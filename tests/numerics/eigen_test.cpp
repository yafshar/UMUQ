#include <iostream>

#include "numerics/eigenmatrix.hpp"
#include "gtest/gtest.h"

// Tests
TEST(eigen_test, HandlesMap)
{
    double *A = nullptr;
    A = new double[12];
    for (int i = 0; i < 12; i++)
    {
        A[i] = (double)i;
    }

    //Copy the buffer to the new Eigen object
    EMatrixXd ACopy = EMapXd(A, 3, 4);

    for (int i = 0, l = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++, l++)
        {
            EXPECT_DOUBLE_EQ(A[l], ACopy(i, j));
        }
    }

    //Map the buffer to an Eigen object format
    TEMapXd AMap(A, 3, 4);
    A[0] = -100.;
    A[5] = 200.;
    A[9] = 900.;
    A[11] = -23.;

    EXPECT_NE(AMap(0, 0), ACopy(0, 0));
    EXPECT_NE(AMap(1, 1), ACopy(1, 1));
    EXPECT_NE(AMap(2, 1), ACopy(2, 1));
    EXPECT_NE(AMap(2, 3), ACopy(2, 3));

    EXPECT_DOUBLE_EQ(AMap(0, 0), -100.);
    EXPECT_DOUBLE_EQ(AMap(0, 1), 1.);
    EXPECT_DOUBLE_EQ(AMap(0, 2), 2.);
    EXPECT_DOUBLE_EQ(AMap(0, 3), 3.);
    EXPECT_DOUBLE_EQ(AMap(1, 0), 4.);
    EXPECT_DOUBLE_EQ(AMap(1, 1), 200.);
    EXPECT_DOUBLE_EQ(AMap(1, 2), 6.);
    EXPECT_DOUBLE_EQ(AMap(1, 3), 7.);
    EXPECT_DOUBLE_EQ(AMap(2, 0), 8.);
    EXPECT_DOUBLE_EQ(AMap(2, 1), 900.);
    EXPECT_DOUBLE_EQ(AMap(2, 2), 10.);
    EXPECT_DOUBLE_EQ(AMap(2, 3), -23.);

    delete[] A;
    A = nullptr;

    EXPECT_EQ(nullptr, A);

    A = new double[625];
    for (int i = 0; i < 625; i++)
    {
        A[i] = 0;
    }

    //create a new 25*25 Eigen Matrix C and initialize to random values
    EMatrixXd C = Eigen::Matrix<double, 25, 25>::Random();

    //copy the matrix C into A
    EMapXd(C, A);

    //check to see if the copy process has been done correctly
    for (int i = 0, l = 0; i < 25; i++)
    {
        for (int j = 0; j < 25; j++, l++)
        {
            EXPECT_DOUBLE_EQ(A[l], C(i, j));
        }
    }

    //change one value in the array
    A[624] = 100.0;

    //check to make sure that the matrix C is a copy of the array buffer
    EXPECT_NE(A[624], C(24, 24));

    //destroy the object
    delete[] A;
    A = nullptr;
};

// Linear Algebra test
TEST(eigen_test, HandlesSolver)
{
    EMatrix6d A;
    A << 1, 1, 0, 1, 0, 0,
        1, 0, 1, 0, 0, 1,
        1, -1, 0, 1, -0, 0,
        1, 0, -1, 0, -0, 1,
        1, 0.70710678118654752440, 0.70710678118654752440, 0.5, 0.5, 0.5,
        1, -0.70710678118654752440, -0.70710678118654752440, 0.5, 0.5, 0.5;

    EVector6d B = EVector6d::Ones();

    std::cout << A << std::endl;

    //LU decomposition of a matrix with complete pivoting
    // Eigen::FullPivLU<EMatrix6d> lu(A);

    EVector6d X = A.fullPivLu().solve(B);

    auto relative_error = (A * X - B).norm() / B.norm();

    EXPECT_DOUBLE_EQ(relative_error, 0);

    X = A.partialPivLu().solve(B);

    relative_error = (A * X - B).norm() / B.norm();

    EXPECT_TRUE(std::isnan(relative_error));
}

// SVD test
TEST(eigen_test, HandlesSVD)
{
    //This is the example from wikipedia
    //https://en.wikipedia.org/wiki/Singular-value_decomposition
    Eigen::Matrix<double, 4, 5> A;
    A << 1, 0, 0, 0, 2,
        0, 0, 3, 0, 0,
        0, 0, 0, 0, 0,
        0, 2, 0, 0, 0;

    // Two-sided Jacobi iterations is numerically very accurate, fast for small matrices, but very slow for larger ones.
    Eigen::JacobiSVD<Eigen::Matrix<double, 4, 5>> svd(A);
    EVector4d B(svd.singularValues());

    EXPECT_DOUBLE_EQ(3, B(0));
    EXPECT_DOUBLE_EQ(std::sqrt(5.), B(1));
    EXPECT_DOUBLE_EQ(2, B(2));
    EXPECT_DOUBLE_EQ(0, B(3));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
