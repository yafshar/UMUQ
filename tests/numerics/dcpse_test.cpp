#include "core/core.hpp"
#include "io/io.hpp"
#include "numerics/dcpse.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check dcpse functionality
 */
TEST(dcpse_test, HandlesDCoperators)
{
    dcpse<double> dc(1);

    int npoints = 10;
    double *idata = new double[npoints];
    int nqpoints = npoints / 2;
    double *qdata = new double[nqpoints];
    double dx = 1. / npoints;
    std::fill(idata, idata + npoints, dx);
    std::partial_sum(idata, idata + npoints, idata, std::plus<double>());

    for (int i = 0; i < npoints, i++)
    {
        std::cout << i << " " << idata[i] << std::endl;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
