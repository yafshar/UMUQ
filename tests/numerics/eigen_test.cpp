#include "core/core.hpp"
#include "numerics/eigenmatrix.hpp"
#include "gtest/gtest.h"

#ifdef HAVE_EIGEN
/*! 
 * Test to check about map type handling is done correctly
 */
TEST(eigen_test, HandlesMap)
{
    double *A = nullptr;
    A = new double[12];
    for (int i = 0; i < 12; i++)
    {
        A[i] = (double)i;
    }

    //!Copy the buffer to the new Eigen object
    EMatrixXd ACopy = EMapXd(A, 3, 4);

    for (int i = 0, l = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++, l++)
        {
            EXPECT_DOUBLE_EQ(A[l], ACopy(i, j));
        }
    }

    //!Map the buffer to an Eigen object format
    TEMapXd AMap(A, 3, 4);
    A[0] = -100.;
    A[1] = 200.0;
    A[5] = 500.0;
    A[9] = 900.0;
    A[11] = -23.;

    EXPECT_NE(AMap(0, 0), ACopy(0, 0));
    EXPECT_NE(AMap(0, 1), ACopy(0, 1));
    EXPECT_NE(AMap(1, 1), ACopy(1, 1));
    EXPECT_NE(AMap(2, 1), ACopy(2, 1));
    EXPECT_NE(AMap(2, 3), ACopy(2, 3));

    EXPECT_DOUBLE_EQ(AMap(0, 0), -100.);
    EXPECT_DOUBLE_EQ(AMap(0, 1), 200.);
    EXPECT_DOUBLE_EQ(AMap(0, 2), 2.);
    EXPECT_DOUBLE_EQ(AMap(0, 3), 3.);
    EXPECT_DOUBLE_EQ(AMap(1, 0), 4.);
    EXPECT_DOUBLE_EQ(AMap(1, 1), 500.);
    EXPECT_DOUBLE_EQ(AMap(1, 2), 6.);
    EXPECT_DOUBLE_EQ(AMap(1, 3), 7.);
    EXPECT_DOUBLE_EQ(AMap(2, 0), 8.);
    EXPECT_DOUBLE_EQ(AMap(2, 1), 900.);
    EXPECT_DOUBLE_EQ(AMap(2, 2), 10.);
    EXPECT_DOUBLE_EQ(AMap(2, 3), -23.);

    AMap(0, 0) = 200.0;
    EXPECT_DOUBLE_EQ(AMap(0, 0), A[0]);

    //!Map the buffer to a read only Eigen object format
    // CTEMapXd CAMap(A, 3, 4);

    delete[] A;
    A = nullptr;

    EXPECT_EQ(nullptr, A);

    A = new double[625];
    for (int i = 0; i < 625; i++)
    {
        A[i] = 0;
    }

    //!create a new 25*25 Eigen Matrix C and initialize to random values
    EMatrixXd C = Eigen::Matrix<double, 25, 25>::Random();

    //!copy the matrix C into A
    EMapXd(C, A);

    //!check to see if the copy process has been done correctly
    for (int i = 0, l = 0; i < 25; i++)
    {
        for (int j = 0; j < 25; j++, l++)
        {
            EXPECT_DOUBLE_EQ(A[l], C(i, j));
        }
    }

    //!change one value in the array
    A[624] = 100.0;

    //!check to make sure that the matrix C is a copy of the array buffer
    EXPECT_NE(A[624], C(24, 24));

    //!destroy the object
    delete[] A;
    A = nullptr;

    A = &C(0, 0);

    A[0] = 10000.0;
    EXPECT_DOUBLE_EQ(C(0, 0), 10000.0);

    double **D;
    D = new double *[3];
    for (int i = 0; i < 3; i++)
    {
        D[i] = new double[3];
    }

    for (int i = 0, l = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++, l++)
        {
            D[i][j] = (double)l;
        }
    }

    EMatrixXd DCopy = EMapXd(D, 3, 3);

    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            EXPECT_DOUBLE_EQ(DCopy(i, j), D[i][j]);
        }
    }

    //!Converts them to an array, which uses to multiply them coefficient-wise
    DCopy = DCopy.array() * DCopy.array();

    EMapXd(DCopy, D);

    for (int i = 0, l = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++, l++)
        {
            EXPECT_DOUBLE_EQ(D[i][j], (double)l * l);
        }
    }

    delete[] * D;
    delete[] D;
};

/*! 
 * Linear Algebra test
 */
TEST(eigen_la_test, HandlesSolver)
{
    EMatrix6d A;
    A << 1, 1, 0, 1, 0, 0,
        1, 0, 1, 0, 0, 1,
        1, -1, 0, 1, -0, 0,
        1, 0, -1, 0, -0, 1,
        1, 0.70710678118654752440, 0.70710678118654752440, 0.5, 0.5, 0.5,
        1, -0.70710678118654752440, -0.70710678118654752440, 0.5, 0.5, 0.5;

    EVector6d B = EVector6d::Ones();

    //!LU decomposition of a matrix with complete pivoting
    //!Eigen::FullPivLU<EMatrix6d> lu(A);

    EVector6d X = A.fullPivLu().solve(B);

    auto relative_error = (A * X - B).norm() / B.norm();

    EXPECT_DOUBLE_EQ(relative_error, 0);

    X = A.partialPivLu().solve(B);

    relative_error = (A * X - B).norm() / B.norm();

    EXPECT_TRUE(std::isnan(relative_error));
}

//! SVD test
TEST(eigen_svd_test, HandlesSVD)
{
    //!This is the example from wikipedia
    //!https://en.wikipedia.org/wiki/Singular-value_decomposition
    Eigen::Matrix<double, 4, 5> A;
    A << 1, 0, 0, 0, 2,
        0, 0, 3, 0, 0,
        0, 0, 0, 0, 0,
        0, 2, 0, 0, 0;

    //!Two-sided Jacobi iterations is numerically very accurate, fast for small matrices, but very slow for larger ones.
    Eigen::JacobiSVD<Eigen::Matrix<double, 4, 5>> svd(A);
    EVector4d B(svd.singularValues());

    EXPECT_DOUBLE_EQ(3, B(0));
    EXPECT_DOUBLE_EQ(std::sqrt((double)(5)), B(1));
    EXPECT_DOUBLE_EQ(2, B(2));
    EXPECT_DOUBLE_EQ(0, B(3));
}

//! LU test
TEST(eigen_lu_test, HandlesLU)
{
    //A 3*3 matrix with rank 2 which is not invertible
    EMatrix3d m;
    m << 1, 1, 0,
        1, 3, 1,
        0, 2, 1;

    //LU decomposition of a matrix with complete pivoting, and related features.
    Eigen::FullPivLU<EMatrix3d> lu(m);

    //the rank of the matrix m with lu decomposition.
    EXPECT_EQ(lu.rank(), 2);

    //false as the matrix m with lu decomposition is not invertible.
    EXPECT_FALSE(lu.isInvertible());

    //column vector
    EVector3d n;
    n << 5, 5, 5;

    //Creating the new matrix from image (also called its column-space) of it and a new vector.
    m << m.fullPivLu().image(m), n;

    //LU decomposition of a matrix with complete pivoting, and related features.
    lu.compute(m);

    //the rank of the matrix m with lu decomposition.
    EXPECT_EQ(lu.rank(), 3);

    //true as the matrix m with lu decomposition is invertible.
    EXPECT_TRUE(lu.isInvertible());


    std::cout << EMatrix3i(EVector3i(2,5,6).asDiagonal()) << std::endl;
}
#endif
#ifndef HAVE_EIGEN
/*! 
 * Test to check about map type handling is done correctly
 */
TEST(eigen_test, HandlesMap)
{
}
#endif

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
