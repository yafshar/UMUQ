#include "core/core.hpp"
#include "numerics/dcpse.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check dcpse functionality
 */
TEST(dcpse_test, HandlesDCoperators)
{
	dcpse<double> dc(2);

	EMatrix3d yaser = EMatrix3d::Zero();
	EVector3d y1, y2, y3;

	y1 << 1, 2, 3;
	y2 << 4, 5, 6;
	y3 << 7, 8, 9;

	yaser << y1, y2, y3;

	std::cout << "-----------" << std::endl;
	std::cout << yaser << std::endl;
	std::cout << "-----------" << std::endl;

	double y4[3] = {100, 200, 300};

	yaser.block(0, 1, 3, 1) << Eigen::Map<EVector3d>(y4);
	std::cout << yaser << std::endl;
	std::cout << "-----------" << std::endl;
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
