#include "inference/prior/priordistribution.hpp"
#include "numerics/random/psrandom.hpp"
#include "environment.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *
 * \brief Get an instance of a seeded pseudo random object
 */
umuq::psrandom prng(12345678);

/*!
 * \ingroup Test_Module
 *
 * \brief uniform PDF
 *
 * \param param1 lower bound
 * \param param2 upper bound
 * \returns  uniform PDF
 */
double priorpdf(std::vector<double> const &param1, std::vector<double> const &param2)
{
    double sum{1};
    for (std::size_t i = 0; i < param2.size(); i++)
    {
        sum *= 1. / (param2[i] - param1[i]);
    }
    return sum;
}

/*!
 * \ingroup Test_Module
 *
 * \brief uniform Log(PDF)
 *
 * \param param1 lower bound
 * \param param2 upper bound
 * \returns uniform Log(PDF)
 */
double logpriorpdf(std::vector<double> const &param1, std::vector<double> const &param2)
{
    double sum{};
    for (std::size_t i = 0; i < param2.size(); i++)
    {
        sum += -std::log(param2[i] - param1[i]);
    }
    return sum;
}

// Tests priorDistribution
TEST(priorDistribution_test, HandlesConstruction)
{
    // uniform prior distribution in 4 dimensions
    umuq::priorDistribution<double> prior(4);

    EXPECT_EQ(prior.getpriorType(), umuq::priorTypes::UNIFORM);

    //              Lower bound    Upper bound
    // B0              0.05           10.0
    // B1              3.0             4.0
    // B2              6.01           15.0
    // B3              0.0001          1.0
    std::vector<double> lowerbound = {0.05, 3.0, 6.01, 0.0001};
    std::vector<double> upperbound = {10.0, 4.0, 15.0, 1.0000};

    // Set the prior
    prior.set(lowerbound, upperbound);

    double x[] = {5, 3.5, 12., 0.001};

    EXPECT_DOUBLE_EQ(prior.pdf(x), priorpdf(lowerbound, upperbound));
    EXPECT_DOUBLE_EQ(prior.logpdf(x), logpriorpdf(lowerbound, upperbound));

    // Initialize the PRNG or set the state of the PRNG
    EXPECT_TRUE(prng.setState());

    for (int i = 0; i < 1000; i++)
    {
        // Sampling from this prior distribution
        EXPECT_TRUE(prior.sample(x));

        EXPECT_TRUE(x[0] >= lowerbound[0] && x[0] <= upperbound[0]);
        EXPECT_TRUE(x[1] >= lowerbound[1] && x[1] <= upperbound[1]);
        EXPECT_TRUE(x[2] >= lowerbound[2] && x[2] <= upperbound[2]);
        EXPECT_TRUE(x[3] >= lowerbound[3] && x[3] <= upperbound[3]);
    }
}

// Tests composite priorDistribution
TEST(priorDistribution_test, HandlesCompositePriorConstruction)
{
    {
        // Composite prior distribution in 2 dimensions
        umuq::priorDistribution<double> prior(2, umuq::priorTypes::COMPOSITE);

        EXPECT_EQ(prior.getpriorType(), umuq::priorTypes::COMPOSITE);

        // Composite  Prior type   Parameter1      Parameter2
        //  C0            1          0.0              1.0
        //  C1            0          0.05            10.0
        std::vector<double> param1 = {0.0, 0.05};
        std::vector<double> param2 = {1.0, 10.0};


        // For the first dimension it is a Gaussian distribution with \f$ \mu=0 \f$ and \f$ \sigma = 1 \f$
        // second dimension has theuniform distribution
        std::vector<umuq::priorTypes> compositeprior = {umuq::priorTypes::GAUSSIAN, umuq::priorTypes::UNIFORM};

        // Set the prior
        prior.set(param1, param2, compositeprior);

        double x[] = {3.5, 5.};

        EXPECT_DOUBLE_EQ(prior.pdf(x), 1 / M_S2PI * std::exp(-x[0] * x[0] / 2.) * priorpdf(std::vector<double>{param1[1]}, std::vector<double>{param2[1]}));
        EXPECT_DOUBLE_EQ(prior.logpdf(x), -0.5 * M_L2PI - x[0] * x[0] / 2. + logpriorpdf(std::vector<double>{param1[1]}, std::vector<double>{param2[1]}));

        // Initialize the PRNG or set the state of the PRNG
        EXPECT_TRUE(prng.setState());

        // Sampling from this prior distribution
        EXPECT_TRUE(prior.sample(x));

        EXPECT_TRUE(x[1] >= param1[1] && x[1] <= param2[1]);
    }

    {
        // Composite prior distribution in 2 dimensions
        umuq::priorDistribution<double> prior(2, 4);

        // Converting the enum class to underlying type for comparison to int
        EXPECT_EQ(static_cast<std::underlying_type_t<umuq::priorTypes>>(prior.getpriorType()), 4);

        // Composite  Prior type   Parameter1      Parameter2
        //  C0            1          0.0              1.0
        //  C1            0          0.05            10.0
        std::vector<double> param1 = {0.0, 0.05};
        std::vector<double> param2 = {1.0, 10.0};

        // For the first dimension it is a Gaussian distribution with \f$ \mu=0 \f$ and \f$ \sigma = 1 \f$
        // second dimension has theuniform distribution
        std::vector<int> compositeprior = {1, 0};

        // Set the prior
        prior.set(param1, param2, compositeprior);

        double x[] = {3.5, 5.};

        EXPECT_DOUBLE_EQ(prior.pdf(x), 1 / M_S2PI * std::exp(-x[0] * x[0] / 2.) * priorpdf(std::vector<double>{param1[1]}, std::vector<double>{param2[1]}));
        EXPECT_DOUBLE_EQ(prior.logpdf(x), -0.5 * M_L2PI - x[0] * x[0] / 2. + logpriorpdf(std::vector<double>{param1[1]}, std::vector<double>{param2[1]}));

        // Initialize the PRNG or set the state of the PRNG
        EXPECT_TRUE(prng.setState());

        // Sampling from this prior distribution
        EXPECT_TRUE(prior.sample(x));

        EXPECT_TRUE(x[1] >= param1[1] && x[1] <= param2[1]);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}
