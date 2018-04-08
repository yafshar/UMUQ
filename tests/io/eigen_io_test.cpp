#include "numerics/eigenmatrix.hpp"
#include "io/io.hpp"
#include "gtest/gtest.h"

template <typename TM>
bool EM_equal(TM A, TM B)
{
	return (A - B).norm() == 0;
}

/*! 
 * Load and Save of an Eigen mtrix from and to a file in Matrix Format
 */
TEST(eigen_io_test, HandlesLoadandSaveinMatrixFormat)
{
	const char *fileName = "eiotmp";

	//!An instance of io class
	io f;

	//! - 1

	//!Create a matrix of size 4*4 and of type double and fill it with random numbers
	EMatrixXd A = Eigen::Matrix<double, 4, 4>::Random();

	//!Create a new matrix B of the same size and type as A
	EMatrix4d B;

	//!Open a file for reading and writing
	if (f.openFile(fileName, f.in | f.out | f.trunc))
	{
		//!Write the matrix in it
		f.saveMatrix<EMatrixXd, Eigen::IOFormat>(A, fmt);

		//!Rewind the file
		f.rewindFile();

		//!Read the matrix from it
		f.loadMatrix<EMatrix4d>(B);

		//!Compare that the matrix A and B are approximately the same within machine precision
		EXPECT_TRUE(A.isApprox(B));

		f.closeFile();
	}

	//! - 2

	//!Create a new matrix of type int and fill it with random number
	EMatrixXi C = Eigen::Matrix<int, 10, 10>::Random();

	//!Create a new matrix of of the same size and type as C
	EMatrixXi D(10, 10);

	//!Open a file for reading and writing
	if (f.openFile(fileName, f.in | f.out | f.trunc))
	{
		//!Write the matrix in it
		f.saveMatrix<EMatrixXi, Eigen::IOFormat>(C, fmt);

		//!Rewind the file
		f.rewindFile();

		//!Read the matrix from it
		f.loadMatrix<EMatrixXi>(D);

		//!Compare the matrices
		EXPECT_PRED2(EM_equal<EMatrixXi>, C, D);

		f.closeFile();
	}

	//!delete the file
	std::remove(fileName);

	//! - 3
	//!Open a file for reading and writing
	if (f.openFile(fileName, f.in | f.out | f.app))
	{
		//!write down two matrices of different types in it
		f.saveMatrix<EMatrixXd, Eigen::IOFormat>(A, fmt);
		f.saveMatrix<EMatrixXi, Eigen::IOFormat>(C, fmt);

		//!Initialize B and D to zero
		B = EMatrix4d::Zero();
		D = Eigen::Matrix<int, 10, 10>::Zero();

		//!Rewind the file
		f.rewindFile();

		f.loadMatrix<EMatrix4d>(B);
		f.loadMatrix<EMatrixXi>(D);

		//!Compare the matrices
		EXPECT_TRUE(A.isApprox(B));
		EXPECT_PRED2(EM_equal<EMatrixXi>, C, D);

		f.closeFile();
	}

	//!delete the file
	std::remove(fileName);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}