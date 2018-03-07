#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <iomanip>

#include "./numerics/polynomial.hpp"
#include "./numerics/eigenmatrix.hpp"
#include "./numerics/flannlib.hpp"
#include "./misc/timer.hpp"
#include "./misc/spawner.hpp"
#include <lapacke.h>

int main()
{

    int d = 2;
    int r = 2;
    int *alpha = NULL;
    polynomial p;
	UMTimer t;

    p.monomial_basis(d, r, alpha);
	t.toc("monomial_basis");

    std::cout << " d =  " << d << std::endl;
    std::cout << " r =  " << r << std::endl;
    std::cout << " i    [] []" << std::endl;

    int i, n;
    n = 0;
    for (i = 0; i < p.binomial_coefficient(d + r, r); i++)
    {
        std::cout << std::setw(3) << i << "   ";
        for (int j = 0; j < d; j++)
        {
            std::cout << std::setw(2) << alpha[n];
            n++;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------" << std::endl;
	
    n = p.binomial_coefficient(d + r, r);

	t.toc("binomial_coefficient");

    double *value=NULL;
    double *x=NULL;

    x = new double[d]; 

    EMatrixXd A;
    A.resize(n, n);

    value = new double[n];

    for (i = 0; i < n; i++)
    {
        switch (i)
        {
        case (0):
            x[0] = 1;
            x[1] = 0;
            break;
        case (1):
            x[0] = 0;
            x[1] = 1;
            break;
        case (2):
            x[0] = -1;
            x[1] = 0;
            break;
        case (3):
            x[0] = 0;
            x[1] = -1;
            break;
        case (4):
            x[0] = .70710678118654752440;
            x[1] = .70710678118654752440;
            break;
        case (5):
            x[0] = -.70710678118654752440;
            x[1] = -.70710678118654752440;
        }

        p.monomial_value(d, r, alpha, x, value);

        for (int j = 0; j < n; j++)
        {
            A(i,j)=value[j];
        }

    }

    delete[] alpha;
    delete[] value;
    delete[] x;

    std::cout << "Here is the matrix A:" << std::endl;
    std::cout << A << std::endl;

    EVectorXd B;
    B.resize(n);
    for (i = 0; i < n; i++) {
        B(i) = 1;
    } 
    std::cout << "Here is the vector B:" << std::endl;
    std::cout << B << std::endl;

    EVectorXd X = A.fullPivLu().solve(B);
    std::cout << "The solution is: " << std::endl;
    std::cout << X << std::endl;

    auto relative_error = (A * X - B).norm() / B.norm(); // norm() is L2 norm
    std::cout << "The relative error is:" << std::endl;
    std::cout << relative_error << std::endl;

    EVectorXd Y = A.partialPivLu().solve(B);
    std::cout << "The solution is: " << std::endl;
    std::cout << Y << std::endl;

    relative_error = (A * Y - B).norm() / B.norm(); // norm() is L2 norm
    std::cout << "The relative error is:" << std::endl;
    std::cout << relative_error << std::endl;
    std::cout << "----------------" << std::endl;

    // EMatrixXd AA;
    // AA.resize(6, 6);
    // AA << 1, 1, 1, 1, 1, 1,
    //     1, 0, -1, 0, .70710678118654752440, -.70710678118654752440,
    //     0, 1, 0, -1, .70710678118654752440, -.70710678118654752440,
    //     1, 0, 1, 0, 0.5, 0.5,
    //     0, 0, 0, 0, 0.5, 0.5,
    //     0, 1, 0, 1, 0.5, 0.5;
    // std::cout << "--------AA--------" << std::endl;
    // std::cout << AA << std::endl;
    // std::cout << "--------AA^T--------" << std::endl;
    // AA.transposeInPlace();
    // std::cout << AA << std::endl;
    // std::cout << "----------------" << std::endl;
    // Eigen::JacobiSVD<EMatrixXd> svd(AA);
    // std::cout << svd.singularValues() << std::endl;
    // std::cout << "----------inverse----------" << std::endl;
    // std::cout << AA.inverse() << std::endl;

    // // note, to understand this part take a look in the MAN pages, at section of parameters.
    // char TRANS = 'N';
    // int INFO = 3;
    // int LDA = 3;
    // int LDB = 3;
    // int NDIM = 3;
    // int NRHS = 1;
    // int IPIV[3];

    // double AAA[9] =
    //     {
    //         1, 2, 3,
    //         2, 3, 4,
    //         3, 4, 1};

    // double BBB[3] =
    //     {
    //         -4,
    //         -1,
    //         -2};
    // // end of declarations

    // std::cout << "compute the LU factorization..." << std::endl
    //           << std::endl;

    // //void LAPACK_dgetrf( lapack_int* m, lapack_int* n, double* a, lapack_int* lda, lapack_int* ipiv, lapack_int *info );
    // LAPACK_dgetrf(&NDIM, &NDIM, AAA, &LDA, IPIV, &INFO);

    // // checks INFO, if INFO != 0 something goes wrong, for more information see the MAN page of dgetrf.
    // if (INFO)
    // {
    //     std::cout << "an error occured : " << INFO << std::endl
    //               << std::endl;
    // }
    // else
    // {
    //     std::cout << "solving the system..." << std::endl
    //               << std::endl;
    //     // void LAPACK_dgetrs( char* trans, lapack_int* n, lapack_int* nrhs, const double* a, lapack_int* lda, const lapack_int* ipiv,double* b, lapack_int* ldb, lapack_int *info );
    //     dgetrs_(&TRANS, &NDIM, &NRHS, AAA, &LDA, IPIV, BBB, &LDB, &INFO);

    //     if (INFO)
    //     {
    //         // checks INFO, if INFO != 0 something goes wrong, for more information see the MAN page of dgetrs.
    //         std::cout << "an error occured : " << INFO << std::endl
    //                   << std::endl;
    //     }
    //     else
    //     {
    //         std::cout << "print the result : {";
    //         for (i = 0; i < NDIM; i++)
    //         {
    //             std::cout << BBB[i] << " ";
    //         }
    //         std::cout << "}" << std::endl
    //                   << std::endl;
    //     }
    // }

    // std::cout << "program terminated." << std::endl
    //           << std::endl;

    return 0;
}
