#include "core/core.hpp"
#include "environment.hpp"
#include "inference/prior/priordistribution.hpp"
#include "gtest/gtest.h"

//! uniform PDF
double priorpdf(std::vector<double> const &lowerbound, std::vector<double> const &upperbound)
{
    double sum{1};
    for (std::size_t i = 0; i < upperbound.size(); i++)
    {
        sum *= 1. / (upperbound[i] - lowerbound[i]);
    }
    return sum;
}

//! uniform Log(PDF)
double logpriorpdf(std::vector<double> const &lowerbound, std::vector<double> const &upperbound)
{
    double sum{};
    for (std::size_t i = 0; i < upperbound.size(); i++)
    {
        sum += -std::log(upperbound[i] - lowerbound[i]);
    }
    return sum;
}

//! Tests priorDistribution
TEST(priorDistribution_test, HandlesConstruction)
{
    //! uniform prior distribution in 4 dimensions
    umuq::priorDistribution<double> prior(4, 0);

    EXPECT_EQ(prior.getpriorType(), 0);

    //              Lower bound    Upper bound
    // B0              0.05           10.0
    // B1              3.0             4.0
    // B2              6.01           15.0
    // B3              0.0001          1.0
    std::vector<double> lowerbound = {0.05, 3.0, 6.01, 0.0001};
    std::vector<double> upperbound = {10.0, 4.0, 15.0, 1.0000};

    //! Set the prior
    prior.set(lowerbound.data(), upperbound.data());

    double x[] = {5, 3.5, 12., 0.001};

    EXPECT_DOUBLE_EQ(prior.pdf(x), priorpdf(lowerbound, upperbound));
    EXPECT_DOUBLE_EQ(prior.logpdf(x), logpriorpdf(lowerbound, upperbound));
}

//! Tests composite priorDistribution
TEST(priorDistribution_test, HandlesCompositePriorConstruction)
{
    //! composite prior distribution in 2 dimensions
    umuq::priorDistribution<double> prior(2, 4);

    EXPECT_EQ(prior.getpriorType(), 4);

    //Composite  Prior type   Parameter1      Parameter2
    // C0            0          0.05            10.0
    // C1            0          3.0              4.0
    std::vector<double> lowerbound = {0.05, 3.0};
    std::vector<double> upperbound = {10.0, 4.0};
    std::vector<int> compositeprior = {0, 0};

    //! Set the prior
    prior.set(lowerbound.data(), upperbound.data(), compositeprior.data());

    double x[] = {5, 3.5};

    EXPECT_DOUBLE_EQ(prior.pdf(x), priorpdf(lowerbound, upperbound));
    EXPECT_DOUBLE_EQ(prior.logpdf(x), logpriorpdf(lowerbound, upperbound));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment<>);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}