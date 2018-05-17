#include "core/core.hpp"
#include "io/io.hpp"
#include "numerics/dcpse.hpp"
#include "gtest/gtest.h"

/*!
 * \brief Franke's bivariate test function
 * 
 * Franke's bivariate test function is a weighted sum of four exponentials
 * \f[
 * f(x) &= 0.75 e^\left(-\frac{(9x_1-2)^2}{4} - \frac{(9x_2-2)^2}{4} \right) \\
 *      &+ 0.75 e^\left(-\frac{(9x_1+1)^2}{49} - \frac{(9x_2+1)}{10} \right) \\
 *      &+ 0.5 e^\left(-\frac{(9x_1-7)^2}{4} - \frac{(9x_2-3)^2}{4} \right) \\
 *      &- 0.2 e^\left(-(9x_1-4)^2 - (9x_2-7)^2 \right)
 * \f]
 * 
 * \tparam T    data type
 * \param idata input data point
 * \return T    function value at input data point
 */
template <typename T>
inline T franke2d(T const *idata)
{
	T const x1 = idata[0];
	T const x2 = idata[1];
	T const term1 = 0.75 * std::exp(-std::pow(9 * x1 - 2, 2) / 4 - std::pow(9 * x2 - 2, 2) / 4);
	T const term2 = 0.75 * std::exp(-std::pow(9 * x1 + 1, 2) / 49 - (9 * x2 + 1) / 10);
	T const term3 = 0.5 * std::exp(-std::pow(9 * x1 - 7, 2) / 4 - std::pow(9 * x2 - 3, 2) / 4);
	T const term4 = -0.2 * std::exp(-std::pow(9 * x1 - 4, 2) - std::pow(9 * x2 - 7, 2));
	return term1 + term2 + term3 + term4;
}

/*! 
 * Test to check dcpse functionality
 */
TEST(dcpse_test, HandlesDCoperators)
{
	dcpse<double> dc(2);

	int npoints = 10;
	double *idata = new double[npoints];
	int nqpoints = npoints / 2;
	double *qdata = new double[nqpoints];
	double dx = 1. / npoints;
	std::fill(idata, idata + npoints, dx);
	std::partial_sum(idata, idata + npoints, idata, std::plus<double>());

	for (int i = 0; i < npoints; i++)
	{
		std::cout << i << " " << idata[i] << std::endl;
	}
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
