#include "core/core.hpp"
#include "numerics/fitness.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check fitness and residual functionality
 */
TEST(fitness_test, HandlesFitness)
{
    umuq::fitness<double> f;
    EXPECT_EQ(f.getMetricName(), "sum_squared");
}

/*! 
 * Test to check fitness and residual functionality in simple linear regression
 */
TEST(fitness_test, HandlesLinearRegression)
{
    int nPoints = 10;

    std::unique_ptr<double[]> idata(new double[nPoints]);
    std::unique_ptr<double[]> observations(new double[nPoints]);
    std::unique_ptr<double[]> predictions(new double[nPoints]);

    // std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double *d = idata.get();
    std::for_each(d, d + nPoints, [&](double &i) { i = dis(gen); });

    d = idata.get();
    double *o = observations.get();
    std::for_each(o, o + nPoints, [&](double &i) { i = (*d) * 2. + 1.; d++; });

    o = observations.get();
    double *p = predictions.get();
    std::for_each(p, p + nPoints, [&](double &i) { i = (*o) + (dis(gen) - 0.5) * 0.01; o++; });

    o = observations.get();
    p = predictions.get();

    umuq::fitness<double> f("sum_squared");
    std::cout << "For " << f.getMetricName() << " : " << f.getFitness(o, p, nPoints) << std::endl;

    f.setMetricName("mean_squared");
    std::cout << "For " << f.getMetricName() << " : " << f.getFitness(o, p, nPoints) << std::endl;

    f.setMetricName("root_mean_squared");
    std::cout << "For " << f.getMetricName() << " : " << f.getFitness(o, p, nPoints) << std::endl;

    f.setMetricName("max_squared");
    std::cout << "For " << f.getMetricName() << " : " << f.getFitness(o, p, nPoints) << std::endl;

    d = idata.get();
    std::iota(d, d + nPoints, double{});

    d = idata.get();
    o = observations.get();
    std::for_each(o, o + nPoints, [&](double &i) { i = (*d) * (*d); d++; });

    d = idata.get();
    p = predictions.get();
    std::for_each(p, p + nPoints, [&](double &i) { i = (*d) * (*d) + 0.1; d++; });

    o = observations.get();
    p = predictions.get();
    f.setMetricName("root_mean_squared");
    EXPECT_NEAR(f.getFitness(o, p, nPoints), 0.1, 1.e-8);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
