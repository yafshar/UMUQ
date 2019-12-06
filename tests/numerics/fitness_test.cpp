#include "numerics/fitness.hpp"
#include "gtest/gtest.h"

#include <random>

/*!
 * \ingroup Test_Module
 *
 * Test to check fitness and residual functionality
 */
TEST(fitness_test, HandlesFitness)
{
    umuq::fitness<double> f;
    EXPECT_EQ(f.getMetricName(), "SUM_SQUARED");
}

/*!
 * \ingroup Test_Module
 *
 * Test to check fitness and residual functionality in simple linear regression
 */
TEST(fitness_test, HandlesLinearRegression)
{
    int nPoints = 10;

    std::vector<double> idata(nPoints);
    std::vector<double> observations(nPoints);
    std::vector<double> predictions(nPoints);

    // std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::for_each(idata.begin(), idata.end(), [&](double &i) { i = dis(gen); });

    {
        auto d = idata.begin();
        std::for_each(observations.begin(), observations.end(), [&](double &i) { i = (*d) * 2. + 1.; d++; });
    }

    {
        auto o = observations.begin();
        std::for_each(predictions.begin(), predictions.end(), [&](double &i) { i = (*o) + (dis(gen) - 0.5) * 0.01; o++; });
    }

    umuq::fitness<double> f("sum_squared");

    {
        auto fitnessValue = f.getFitness(observations.data(), predictions.data(), nPoints);
        EXPECT_DOUBLE_EQ(f.getFitness(observations, predictions), fitnessValue);
        std::cout << "For " << f.getMetricName() << " : " << fitnessValue << std::endl;
    }

    {
        f.setMetricName("mean_squared");
        auto fitnessValue = f.getFitness(observations.data(), predictions.data(), nPoints);
        EXPECT_DOUBLE_EQ(f.getFitness(observations, predictions), fitnessValue);
        std::cout << "For " << f.getMetricName() << " : " << fitnessValue << std::endl;
    }

    {
        f.setMetricName("root_mean_squared");
        auto fitnessValue = f.getFitness(observations.data(), predictions.data(), nPoints);
        EXPECT_DOUBLE_EQ(f.getFitness(observations, predictions), fitnessValue);
        std::cout << "For " << f.getMetricName() << " : " << fitnessValue << std::endl;
    }

    {
        f.setMetricName("max_squared");
        auto fitnessValue = f.getFitness(observations.data(), predictions.data(), nPoints);
        EXPECT_DOUBLE_EQ(f.getFitness(observations, predictions), fitnessValue);
        std::cout << "For " << f.getMetricName() << " : " << fitnessValue << std::endl;
    }

    std::iota(idata.begin(), idata.end(), double{});

    {
        auto d = idata.begin();
        std::for_each(observations.begin(), observations.end(), [&](double &i) { i = (*d) * (*d); d++; });
    }

    {
        auto d = idata.begin();
        std::for_each(predictions.begin(), predictions.end(), [&](double &i) { i = (*d) * (*d) + 0.1; d++; });
    }

    f.setMetricName("root_mean_squared");
    EXPECT_NEAR(f.getFitness(observations.data(), predictions.data(), nPoints), 0.1, 1.e-8);
    EXPECT_NEAR(f.getFitness(observations, predictions), 0.1, 1.e-8);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
