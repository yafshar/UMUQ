#include <iostream>
#include <fstream>
#include <ios>

#include "numerics/eigenmatrix.hpp"
#include "gtest/gtest.h"

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

template <typename TEMX>
bool EM_equal(TEMX A, TEMX B)
{
    return (A - B).norm() == 0;
}

/*! 
 * Load and Save of an Eigen mtrix from and to a file 
 */
TEST(eigen_io_test, HandlesLoadandSave)
{

    const char *fileName = "tmp";
    std::fstream fs;

    //! - 1
    {
        //!Create a matrix of size 4*4 and of type double and fill it with random numbers
        EMatrixXd A = Eigen::Matrix<double, 4, 4>::Random();

        //!Create a new matrix B of the same size and type as A
        EMatrix4d B;

        //!Open a file for reading and writing
        fs.open(fileName, std::fstream::in | std::fstream::out | std::fstream::trunc);

        //!Write the matrix in it
        saveMatrix<EMatrixXd>(fs, A);

        //!Rewind the file
        fs.seekg(0);

        //!Read the matrix from it
        loadMatrix<EMatrix4d>(fs, B);

        fs.close();

        //!Compare that the matrix A and B are approximately the same within machine precision
        EXPECT_TRUE(A.isApprox(B));

        //! - 2

        //!Create a new matrix of type int and fill it with random number
        EMatrixXi C = Eigen::Matrix<int, 10, 10>::Random();

        //!Create a new matrix of of the same size and type as C
        EMatrixXi D(10, 10);

        //!Open a file for reading and writing
        fs.open(fileName, std::fstream::in | std::fstream::out | std::fstream::trunc);

        //!Write the matrix in it
        saveMatrix<EMatrixXi>(fs, C);

        //!Rewind the file
        fs.seekg(0);

        //!Read the matrix from it
        loadMatrix<EMatrixXi>(fs, D);

        fs.close();
        //!delete the file
        std::remove(fileName);

        //!Compare the matrices
        EXPECT_PRED2(EM_equal<EMatrixXi>, C, D);

        //! - 3

        //!Open a file for reading and writing
        fs.open(fileName, std::fstream::in | std::fstream::out | std::fstream::app);

        //!write down two matrices of different types in it
        saveMatrix<EMatrixXd>(fs, A);
        saveMatrix<EMatrixXi>(fs, C);

        //!Initialize B and D to zero
        B = EMatrix4d::Zero();
        D = Eigen::Matrix<int, 10, 10>::Zero();

        //!Rewind the file
        fs.seekg(0);

        loadMatrix<EMatrix4d>(fs, B);
        loadMatrix<EMatrixXi>(fs, D);

        fs.close();

        //!Compare the matrices
        EXPECT_TRUE(A.isApprox(B));
        EXPECT_PRED2(EM_equal<EMatrixXi>, C, D);
    }

    //! - 4
    {
        //!Create a new array and initialize it
        int *E = new int[12];
        for (int i = 0; i < 12; i++)
        {
            E[i] = i;
        }

        //!Create a new array
        int *F = new int[12];

        //!Open a file for reading and writing
        fs.open(fileName, std::fstream::in | std::fstream::out | std::fstream::trunc);

        //!Save the array in a matrix format
        saveMatrix<int>(fs, E, 3, 4);

        //!Rewind the file
        fs.seekg(0);

        //!Read the array
        loadMatrix<int>(fs, F, 3, 4);

        fs.close();

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                EXPECT_EQ(F[i], E[i]);
            }
        }

        delete[] E;
        delete[] F;
    }

    //! - 5
    {
        //!Create a new array and initialize it
        double **G = nullptr;
        G = new double *[3];
        for (int i = 0; i < 3; i++)
        {
            G[i] = new double[4];
        }
        for (int i = 0, l = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++, l++)
            {
                G[i][j] = (double)l;
            }
        }

        //!Create a new array
        double H[3][4];

        //!Open a file for reading and writing
        fs.open(fileName, std::fstream::in | std::fstream::out | std::fstream::trunc);

        //!Write the matrix
        saveMatrix<double>(fs, G, 3, 4);

        //!Rewind the file
        fs.seekg(0);

        //!Read the matrix
        loadMatrix<double>(fs, reinterpret_cast<double *>(H), 3, 4);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                EXPECT_DOUBLE_EQ(H[i][j], G[i][j]);
            }
        }

        fs.close();
        //!delete the file
        std::remove(fileName);

        delete[] * G;
        delete[] G;
    }

    //! - 6
    {
        //!Create a new array and initialize it
        int *I = new int[12];
        for (int i = 0; i < 12; i++)
        {
            I[i] = i;
        }

        //!Open a file for reading and writing
        fs.open(fileName, std::fstream::in | std::fstream::out | std::fstream::trunc);

        //!Write the matrix
        saveMatrix<int>(fs, I, 12);

        //!Rewind the file
        fs.seekg(0);

        int J[12];

        //!Read the matrix
        loadMatrix<int>(fs, J, 12);

        for (int i = 0; i < 12; i++)
        {
            EXPECT_EQ(J[i], I[i]);
        }

        fs.close();
        //!delete the file
        std::remove(fileName);

        delete[] I;
    }

    //! - 7
    {
        //!Create a new array and initialize it
        double **K = nullptr;
        K = new double *[3];
        for (int i = 0; i < 3; i++)
        {
            K[i] = new double[8];
        }
        for (int i = 0, l = 0; i < 3; i++)
        {
            for (int j = 0; j < 8; j++, l++)
            {
                K[i][j] = (double)l;
            }
        }

        //!Create a new array and initialize it
        int *L = new int[20];
        for (int i = 0; i < 20; i++)
        {
            L[i] = i;
        }

        //!Open a file for reading and writing
        fs.open(fileName, std::fstream::in | std::fstream::out | std::fstream::trunc);

        //!Write the matrices
        saveMatrix<double>(fs, K, 3, 8);
        saveMatrix<int>(fs, L, 20);

        //!Rewind the file
        fs.seekg(0);

        double M[3][8];
        int N[20];

        loadMatrix<double>(fs, reinterpret_cast<double *>(M), 3, 8);
        loadMatrix<int>(fs, N, 20);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                EXPECT_DOUBLE_EQ(M[i][j], K[i][j]);
            }
        }

        for (int i = 0; i < 20; i++)
        {
            EXPECT_EQ(N[i], L[i]);
        }

        fs.close();
        //!delete the file
        std::remove(fileName);

        delete[] * K;
        delete[] K;
        delete[] L;
    }

    //! - 8
    {
        //!Create a new array and initialize it
        double **K = nullptr;
        K = new double *[3];
        for (int i = 0; i < 3; i++)
        {
            K[i] = new double[6];
        }
        for (int i = 0, l = 0; i < 3; i++)
        {
            for (int j = 0; j < 6; j++, l++)
            {
                K[i][j] = (double)l;
            }
        }

        //!Open a file for reading and writing
        fs.open("yaser", std::fstream::in | std::fstream::out | std::fstream::trunc);

        //!Write the matrices
        saveMatrix<double>(fs, K, 3, 6);

        double **M = nullptr;
        M = new double *[3];
        for (int i = 0; i < 3; i++)
        {
            M[i] = new double[6];
        }

        //!Rewind the file
        fs.seekg(0);

        loadMatrix<double>(fs, M, 3, 6);

        fs.close();
        //!delete the file
        // std::remove(fileName);

        delete[] * K;
    }

    //! - 9
    {
        // struct ebasic
        // {
        //     double *Parray;
        //     int ndimParray;
        //     double *Garray;
        //     int ndimGarray;
        //     double Fvalue;
        //     int surrogate;
        //     int nsel;
        //     /*!
        //      *  \brief constructor for the default variables
        //      *
        //      */
        //     ebasic() : Parray(NULL),
        //                ndimParray(0),
        //                Garray(NULL),
        //                ndimGarray(0),
        //                Fvalue(0),
        //                surrogate(0),
        //                nsel(0){};
        // };

        // class edatabase
        // {
        //   public:
        //     ebasic *entry;
        //     int entries;
        //     edatabase() : entry(NULL),
        //                   entries(0){};
        // };

        // //!Create data and initialize it
        // edatabase dd;

        // dd.entries = 4;
        // dd.entry = new ebasic[dd.entries];
        // for (int i = 0, l = 0; i < dd.entries; i++, l++)
        // {
        //     dd.entry[i].ndimParray = 2;
        //     dd.entry[i].Parray = new double[dd.entry[i].ndimParray];
        //     for (int j = 0; j < dd.entry[i].ndimParray; j++)
        //     {
        //         dd.entry[i].Parray[j] = (double)(l * l);
        //     }
        //     dd.entry[i].ndimGarray = 4;
        //     dd.entry[i].Garray = new double[dd.entry[i].ndimGarray];
        //     for (int j = 0; j < dd.entry[i].ndimGarray; j++)
        //     {
        //         dd.entry[i].Garray[j] = (double)(l * l * l);
        //     }
        //     dd.entry[i].Fvalue = (double)l;
        // }

        // //!Open a file for reading and writing
        // fs.open(fileName, std::fstream::in | std::fstream::out | std::fstream::trunc);

        // double **tmp = nullptr;
        // tmp = new double *[3];

        // for (int i = 0; i < dd.entries; i++)
        // {
        //     tmp[0] = dd.entry[i].Parray;
        //     tmp[1] = &dd.entry[i].Fvalue;
        //     tmp[2] = dd.entry[i].Garray;

        //     saveMatrix<double>(fs, tmp, 3, )
        // }
    }
}

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
    EXPECT_DOUBLE_EQ(std::sqrt(5.), B(1));
    EXPECT_DOUBLE_EQ(2, B(2));
    EXPECT_DOUBLE_EQ(0, B(3));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
