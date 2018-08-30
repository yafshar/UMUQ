#include "core/core.hpp"
#include "environment.hpp"
#include "numerics/polynomial.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/knearestneighbors.hpp"
#include "misc/timer.hpp"
#include "misc/utility.hpp"
#include <random>

#include "inference/tmcmc/tmcmc.hpp"
#include "numerics/function/functionminimizer.hpp"

int main(int argc, char **argv)
{
	torc_init(argc, argv, 0);

	torc_finalize();

	// int d = 2;
	// int r = 2;

	// int *alpha;

	// polynomial<double> p(d, r);

	// UMTimer t;

	// alpha = p.monomial_basis();
	// t.toc("monomial_basis");

	// std::cout << " d =  " << d << std::endl;
	// std::cout << " r =  " << r << std::endl;
	// std::cout << "  i   [] []" << std::endl;

	// int i, n;
	// n = 0;
	// for (i = 0; i < p.binomial_coefficient(d + r, r); i++)
	// {
	//     std::cout << std::setw(3) << i << "   ";
	//     for (int j = 0; j < d; j++)
	//     {
	//         std::cout << std::setw(2) << alpha[n];
	//         n++;
	//     }
	//     std::cout << std::endl;
	// }
	// std::cout << "----------------" << std::endl;

	// n = p.binomial_coefficient(d + r, r);

	// t.toc("binomial_coefficient");

	// double *value = nullptr;
	// double *x = nullptr;

	// EMatrixXd A;
	// A.resize(n, n);

	// x = new double[d];
	// value = new double[n];

	// for (i = 0; i < n; i++)
	// {
	//     switch (i)
	//     {
	//     case (0):
	//         x[0] = 1;
	//         x[1] = 0;
	//         break;
	//     case (1):
	//         x[0] = 0;
	//         x[1] = 1;
	//         break;
	//     case (2):
	//         x[0] = -1;
	//         x[1] = 0;
	//         break;
	//     case (3):
	//         x[0] = 0;
	//         x[1] = -1;
	//         break;
	//     case (4):
	//         x[0] = .70710678118654752440;
	//         x[1] = .70710678118654752440;
	//         break;
	//     case (5):
	//         x[0] = -.70710678118654752440;
	//         x[1] = -.70710678118654752440;
	//     }

	//     if (p.monomial_value(x, value))
	//     {

	//         for (int j = 0; j < n; j++)
	//         {
	//             A(i, j) = value[j];
	//         }
	//     }
	//     else
	//     {
	//         std::exit(1);
	//     }
	// }

	// delete[] value;
	// delete[] x;
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
