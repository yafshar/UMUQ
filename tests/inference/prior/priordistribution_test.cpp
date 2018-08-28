#include "core/core.hpp"
#include "core/environment.hpp"
#include "inference/prior/priordistribution.hpp"
#include "gtest/gtest.h"

//! uniform
double logpriorpdf(std::vector<double> const &lowerbound, std::vector<double> const &upperbound)
{
    double res{};
    for (std::size_t i = 0; i < upperbound.size(); i++)
    {
        res += -std::log(upperbound[i] - lowerbound[i]);
    }
    return res;
}

//! Tests priorDistribution
TEST(priorDistribution_test, HandlesConstruction)
{
    //! uniform prior distribution in 4 dimensions
    priorDistribution<double> prior(4, 0);

    EXPECT_EQ(prior.getpriorType(), 0);

    std::cout << prior.getpriorType() << std::endl;

    // B0              0.05           10.0
    // B1              3.0             4.0
    // B2              6.01           15.0
    // B3              0.0001          1.0
    std::vector<double> lowerbound = {0.05, 3.0, 6.01, 0.0001};
    std::vector<double> upperbound = {10.0, 4.0, 15.0, 1.0000};
    double x[4];

    prior.set(lowerbound.data(), upperbound.data());

    // EXPECT_DOUBLE_EQ(prior.logpdf(x), logpriorpdf(lowerbound, upperbound));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}