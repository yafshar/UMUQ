#include "core/core.hpp"
#include "numerics/eigenlib.hpp"
#include "gtest/gtest.h"

/*! 
 * \ingroup Test_Module
 * 
 * Test to check about map type handling is done correctly
 */
TEST(eigen_test, HandlesMap)
{
	//! Create an array of data
	std::vector<double> A(12);

	//! Initialize the array
	std::iota(A.begin(), A.end(), double{});

	//! Copy the buffer to the new Eigen object
	umuq::EMatrixXd ACopy = umuq::EMapType<double>(A.data(), 3, 4);

	//! Check to make sure the values are the same
	for (int i = 0, l = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++, l++)
		{
			EXPECT_DOUBLE_EQ(A[l], ACopy(i, j));
		}
	}

	//! Map the buffer to an Eigen object format no copy
	umuq::EMapType<double> AMap(A.data(), 3, 4);

	//! Change some of the values in the original buffer
	A[0] = -100.;
	A[1] = 200.0;
	A[5] = 500.0;
	A[9] = 900.0;
	A[11] = -23.;

	//! Compare the Copied buffer and original data
	EXPECT_NE(AMap(0, 0), ACopy(0, 0));
	EXPECT_NE(AMap(0, 1), ACopy(0, 1));
	EXPECT_NE(AMap(1, 1), ACopy(1, 1));
	EXPECT_NE(AMap(2, 1), ACopy(2, 1));
	EXPECT_NE(AMap(2, 3), ACopy(2, 3));

	//! Make sure that the mapped data is the same as original buffer
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

	//! Check the mutual exclusive
	AMap(0, 0) = 200.0;
	EXPECT_DOUBLE_EQ(AMap(0, 0), A[0]);

	//! Destroy the buffer
	A.clear();
	A.shrink_to_fit();

	EXPECT_EQ(nullptr, A.data());

	//! Allocate memory and initialize to 0
	A.resize(625, double{});

	//! Create a new 25*25 Eigen Matrix C and initialize to random values
	umuq::EMatrixXd C = Eigen::Matrix<double, 25, 25>::Random();

	//! Copy the matrix C into A
	umuq::EMap<umuq::EMatrixXd>(A.data(), C);

	//! Check to see if the copy process has been done correctly
	for (int i = 0, l = 0; i < 25; i++)
	{
		for (int j = 0; j < 25; j++, l++)
		{
			EXPECT_DOUBLE_EQ(A[l], C(i, j));
		}
	}

	//! Change one value in the array
	A[624] = 100.0;

	//! Check to make sure that the matrix C is a copy of the array buffer
	EXPECT_NE(A[624], C(24, 24));

	//! Destroy the buffer
	A.clear();
	A.shrink_to_fit();

	auto *Apointer = C.data();
	Apointer[0] = 10000.0;
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

	umuq::EMatrixXd DCopy = umuq::EMap<umuq::EMatrixXd>(D, 3, 3);

	for (int j = 0; j < 3; j++)
	{
		for (int i = 0; i < 3; i++)
		{
			EXPECT_DOUBLE_EQ(DCopy(i, j), D[i][j]);
		}
	}

	//! Converts them to an array, which uses to multiply them coefficient-wise
	DCopy = DCopy.array() * DCopy.array();

	umuq::EMap<umuq::EMatrixXd>(D, DCopy);

	for (int i = 0, l = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++, l++)
		{
			EXPECT_DOUBLE_EQ(D[i][j], (double)l * l);
		}
	}

	delete[] * D;
	delete[] D;
}

/*!
 * \ingroup Test_Module
 * 
 * Linear Algebra test
 */
TEST(eigen_la_test, HandlesSolver)
{
	umuq::EMatrix6d A;
	A << 1, 1, 0, 1, 0, 0,
		1, 0, 1, 0, 0, 1,
		1, -1, 0, 1, -0, 0,
		1, 0, -1, 0, -0, 1,
		1, 0.70710678118654752440, 0.70710678118654752440, 0.5, 0.5, 0.5,
		1, -0.70710678118654752440, -0.70710678118654752440, 0.5, 0.5, 0.5;

	umuq::EVector6d B = umuq::EVector6d::Ones();

	//! LU decomposition of a matrix with complete pivoting
	//! Eigen::FullPivLU<EMatrix6d> lu(A);

	umuq::EVector6d X = A.fullPivLu().solve(B);

	auto relative_error = (A * X - B).norm() / B.norm();

	EXPECT_DOUBLE_EQ(relative_error, 0);

	X = A.partialPivLu().solve(B);

	relative_error = (A * X - B).norm() / B.norm();

	EXPECT_TRUE(std::isnan(relative_error));
}

/*! 
 * \ingroup Test_Module
 * 
 * \brief SVD test
 * 
 */
TEST(eigen_svd_test, HandlesSVD)
{
	//! This is the example from wikipedia
	//! https://en.wikipedia.org/wiki/Singular-value_decomposition
	Eigen::Matrix<double, 4, 5> A;
	A << 1, 0, 0, 0, 2,
		0, 0, 3, 0, 0,
		0, 0, 0, 0, 0,
		0, 2, 0, 0, 0;

	//!Two-sided Jacobi iterations is numerically very accurate, fast for small matrices, but very slow for larger ones.
	Eigen::JacobiSVD<Eigen::Matrix<double, 4, 5>> svd(A);
	umuq::EVector4<double> B(svd.singularValues());

	EXPECT_DOUBLE_EQ(3, B(0));
	EXPECT_DOUBLE_EQ(std::sqrt((double)(5)), B(1));
	EXPECT_DOUBLE_EQ(2, B(2));
	EXPECT_DOUBLE_EQ(0, B(3));
}

/*!
 * \ingroup Test_Module
 * 
 * \brief LU test
 * 
 */
TEST(eigen_lu_test, HandlesLU)
{
	typedef umuq::EMatrix3<double> EMatrix3d;
	typedef umuq::EVector3<double> EVector3d;

	//! A 3*3 matrix with rank 2 which is not invertible
	EMatrix3d m;
	m << 1, 1, 0,
		1, 3, 1,
		0, 2, 1;

	//! LU decomposition of a matrix with complete pivoting, and related features.
	Eigen::FullPivLU<EMatrix3d> lu(m);

	//! The rank of the matrix m with lu decomposition.
	EXPECT_EQ(lu.rank(), 2);

	//! False as the matrix m with lu decomposition is not invertible.
	EXPECT_FALSE(lu.isInvertible());

	//! Column vector
	EVector3d n;
	n << 5, 5, 5;

	// NOTE :
	// On travis ci OSX system there is this trange bug which would break this code so this is the temporary hack
	{
		Eigen::Matrix<double, 3, 2> p = lu.image(m);

		std::cout << std::endl;
		std::cout << p << std::endl;

		//! Creating the new matrix from image (also called its column-space) of it and a new vector.
		m << p, n;

		std::cout << std::endl;
		std::cout << m << std::endl;
		std::cout << std::endl;

		//! LU decomposition of a matrix with complete pivoting, and related features.
		lu.compute(m);

		//! The rank of the matrix m with lu decomposition.
		EXPECT_EQ(lu.rank(), 3);

		//! True as the matrix m with lu decomposition is invertible.
		EXPECT_TRUE(lu.isInvertible());
	}
}

/*!
 * \ingroup Test_Module
 * 
 * \brief test to check if the matrix is positive definite
 * 
 */
TEST(eigen_PositiveDefinite_test, HandlesIsPositiveDefinite)
{
	//! Matrix A is selfadjoint
	umuq::EMatrix2d A;
	A << 2, 2, 2, 2;

	//! Matrix A is not positive definite
	EXPECT_FALSE(umuq::isSelfAdjointMatrixPositiveDefinite<umuq::EMatrix2d>(A));

	//! Force the matrix to be positive definite
	umuq::forceSelfAdjointMatrixPositiveDefinite<umuq::EMatrix2d>(A);

	//! Check to see if it is positive definite
	EXPECT_TRUE(umuq::isSelfAdjointMatrixPositiveDefinite<umuq::EMatrix2d>(A));

	//! Matrix B is selfadjoint
	std::vector<double> B{1, 1, 3,
						  1, 3, 1,
						  3, 1, 1};

	//! Matrix B is not positive definite
	EXPECT_FALSE(umuq::isSelfAdjointMatrixPositiveDefinite<double>(B.data(), 3));

	//! Force the matrix to be positive definite
	umuq::forceSelfAdjointMatrixPositiveDefinite<double>(B.data(), 3);

	//! Check to see if it is positive definite
	EXPECT_TRUE(umuq::isSelfAdjointMatrixPositiveDefinite<double>(B.data(), 3));
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
