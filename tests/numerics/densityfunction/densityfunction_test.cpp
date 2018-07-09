#include "core/core.hpp"
#include "core/environment.hpp"
#include "numerics/factorial.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/densityfunction/uniformdistribution.hpp"
#include "numerics/densityfunction/exponentialdistribution.hpp"
#include "numerics/densityfunction/gammadistribution.hpp"
#include "numerics/densityfunction/gaussiandistribution.hpp"
#include "numerics/densityfunction/multivariategaussiandistribution.hpp"
#include "io/pyplot.hpp"
#include "gtest/gtest.h"

// Create a global instance of the Pyplot from Pyplot library
pyplot plt;

/*! 
 * Test to check densityFunction functionality
 */
TEST(densityFunction_test, HandlesConstruction)
{
    uniformDistribution<double> u(1, 2);
    std::cout << "uniformDistribution=              " << std::cout.width(32) << u.f(1.5) << "  " << std::cout.width(32) << u.f(3) << std::endl;
    EXPECT_DOUBLE_EQ(u.f(3), double{});

    exponentialDistribution<float> e(1);
    std::cout << "exponentialDistribution=          " << std::cout.width(32) << e.f(1.5) << "  " << std::cout.width(32) << e.f(3) << std::endl;

    gammaDistribution<double> g(0.5);
    std::cout << "gammaDistribution=                " << std::cout.width(32) << g.f(1.5) << "  " << std::cout.width(20) << g.f(3) << std::endl;

    gaussianDistribution<double> gu(2, 5);
    std::cout << "gaussianDistribution=             " << std::cout.width(20) << gu.f(1.5) << "  " << std::cout.width(20) << gu.f(3) << std::endl;

    multivariateGaussianDistribution<double> mvn(2);
    std::cout << "multivariateGaussianDistribution= " << std::cout.width(20) << mvn.f(std::vector<double>{1.5, 2}.data()) << "  " << std::cout.width(20) << mvn.f(std::vector<double>{3, 2}.data()) << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}