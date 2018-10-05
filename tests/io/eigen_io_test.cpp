#include "core/core.hpp"
#include "numerics/eigenlib.hpp"
#include "io/io.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 * 
 * \brief Check the equality of Matrix A & B  
 * 
 * \tparam TM Matrix type
 * 
 * \param A Matrix A 
 * \param B Matrix B
 * 
 * \returns true 
 * \returns false If A != B
 */
template <typename TM>
bool EM_equal(TM A, TM B)
{
	return (A - B).norm() == 0;
}

/*! 
 * \ingroup Test_Module
 * 
 * Load and Save of an Eigen matrix from and to a file in Matrix Format
 */
TEST(eigen_io_test, HandlesLoadandSaveinMatrixFormat)
{
	const char *fileName = "eiotmp";

	//!An instance of io class
	umuq::io f;

	//! - 1

	//!Create a matrix of size 4*4 and of type double and fill it with random numbers
	umuq::EMatrixXd A = Eigen::Matrix<double, 4, 4>::Random();

	//!Create a new matrix B of the same size and type as A
	umuq::EMatrix4<double> B;

	//!Open a file for reading and writing
	if (f.openFile(fileName, f.in | f.out | f.trunc))
	{
		//!Write the matrix in it
		f.saveMatrix<umuq::EMatrixXd, Eigen::IOFormat>(A, umuq::eigenIOFormat);

		//!Rewind the file
		f.rewindFile();

		//!Read the matrix from it
		f.loadMatrix<umuq::EMatrix4<double>>(B);

		//!Compare that the matrix A and B are approximately the same within machine precision
		EXPECT_TRUE(A.isApprox(B));

		f.closeFile();
	}

	//! - 2

	//! Create a new matrix of type int and fill it with random number
	umuq::EMatrixX<int> C = Eigen::Matrix<int, 10, 10>::Random();

	//! Create a new matrix of of the same size and type as C
	umuq::EMatrixX<int> D(10, 10);

	//! Open a file for reading and writing
	if (f.openFile(fileName, f.in | f.out | f.trunc))
	{
		//! Write the matrix in it
		f.saveMatrix<umuq::EMatrixX<int>, Eigen::IOFormat>(C, umuq::eigenIOFormat);

		//! Rewind the file
		f.rewindFile();

		//! Read the matrix from it
		f.loadMatrix<umuq::EMatrixX<int>>(D);

		//! Compare the matrices
		EXPECT_PRED2(EM_equal<umuq::EMatrixX<int>>, C, D);

		f.closeFile();
	}

	//!delete the file
	std::remove(fileName);

	//! - 3
	//! Open a file for reading and writing
	if (f.openFile(fileName, f.in | f.out | f.app))
	{
		//! write down two matrices of different types in it
		f.saveMatrix<umuq::EMatrixX<double>, Eigen::IOFormat>(A, umuq::eigenIOFormat);
		f.saveMatrix<umuq::EMatrixX<int>, Eigen::IOFormat>(C, umuq::eigenIOFormat);

		//! Initialize B and D to zero
		B = umuq::EMatrix4<double>::Zero();
		D = Eigen::Matrix<int, 10, 10>::Zero();

		//! Rewind the file
		f.rewindFile();

		f.loadMatrix<umuq::EMatrix4<double>>(B);
		f.loadMatrix<umuq::EMatrixX<int>>(D);

		//! Compare the matrices
		EXPECT_TRUE(A.isApprox(B));
		EXPECT_PRED2(EM_equal<umuq::EMatrixX<int>>, C, D);

		f.closeFile();
	}

	//! Delete the file
	std::remove(fileName);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}