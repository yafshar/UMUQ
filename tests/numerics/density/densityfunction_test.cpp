#include "core/core.hpp"
#include "environment.hpp"
#include "global.hpp"
#include "numerics/density.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *  
 * Test to check uniformDistribution 
 */
TEST(densityFunction_test, HandlesUniformDistributionConstruction)
{
    // Uniform distribution between 1 and 2
    umuq::density::uniformDistribution<double> u(1, 2);

    double X1 = 1.5;
    double X2 = 3.;

    //
    EXPECT_DOUBLE_EQ(u.f(&X1), 1.);
    EXPECT_DOUBLE_EQ(u.f(&X2), 0.);
    EXPECT_DOUBLE_EQ(u.lf(&X1), 0.);

    // Initialize the PRNG or set the state of the PRNG
    EXPECT_TRUE(umuq::prng.setState());

    // Produce samples with uniform distribution density
    u.sample(&X1);

    std::vector<double> X(10, -1);
    double *p = X.data();
    for (int i = 0; i < 10; i++)
    {
        u.sample(p++);
        EXPECT_TRUE(X[i] >= 1.0 && X[i] <= 2.0);
    }

    // Check for sampling vector
    std::vector<double> Y(10, -1);
    u.sample(Y, 10);

    for (int i = 0; i < 10; i++)
    {
        EXPECT_TRUE(Y[i] >= 1.0 && Y[i] <= 2.0);
    }

    // Check for sampling matrix
    umuq::EMatrixX<double> Z(1, 25);
    u.sample(Z);

    for (int i = 0; i < 25; i++)
    {
        EXPECT_TRUE(Z(0, i) >= 1.0 && Z(0, i) <= 2.0);
    }
}

/*! 
 * \ingroup Test_Module
 * 
 * Test to check exponentialDistribution 
 */
TEST(densityFunction_test, HandlesExponentialDistributionConstruction)
{
    // Exponential distribution with mean 1
    umuq::density::exponentialDistribution<float> e(1);
    float X1 = 1.5f;
    float X2 = 3.f;

    EXPECT_FLOAT_EQ(e.f(&X1), std::exp(-X1));
    EXPECT_FLOAT_EQ(e.f(&X2), std::exp(-X2));
    EXPECT_FLOAT_EQ(e.lf(&X2), -X2);

    // Initialize the PRNG or set the state of the PRNG
    EXPECT_TRUE(umuq::prng.setState());

    // Produce samples with Exponential distribution density
    e.sample(&X1);

    // Check for sampling vector
    std::vector<float> Y(10, -1);

    e.sample(Y, 10);

    // Check for sampling matrix
    umuq::EMatrixX<float> Z(1, 25);

    e.sample(Z);
}

/*! 
 * \ingroup Test_Module
 * 
 * Test to check gammaDistribution 
 */
TEST(densityFunction_test, HandlesGammaDistributionConstruction)
{
    // Gamma distribution with Shape parameter of 0.5
    umuq::density::gammaDistribution<double> g(0.5);
    double X1 = 1.5;
    double X2 = 3.;

    // From MATLAB gampdf(X1, 0.5, 1)
    EXPECT_DOUBLE_EQ(g.f(&X1), 0.10278688653584618);
    // From MATLAB gampdf(X2, 0.5, 1)
    EXPECT_DOUBLE_EQ(g.f(&X2), 0.01621739110988048);
    EXPECT_DOUBLE_EQ(g.lf(&X2), std::log(g.f(&X2)));

    // Initialize the PRNG or set the state of the PRNG
    EXPECT_TRUE(umuq::prng.setState());

    // Produce samples with Exponential distribution density
    g.sample(&X1);

    // Check for sampling vector
    std::vector<double> Y(10, -1);

    g.sample(Y, 10);

    // Check for sampling matrix
    umuq::EMatrixX<double> Z(1, 25);

    g.sample(Z);
}

/*! 
 * \ingroup Test_Module
 * 
 * Test to check gaussianDistribution 
 */
TEST(densityFunction_test, HandlesGaussianDistributionConstruction)
{
    // Gaussian distribution with mean 2 and standard deviation of 5
    umuq::density::gaussianDistribution<double> gu(2, 5);
    double X1 = 1.5;
    double X2 = 3.;

    // From MATLAB normpdf(X1,2,5)
    EXPECT_DOUBLE_EQ(gu.f(&X1), 0.079390509495402356);
    // From MATLAB normpdf(X2, 2, 5)
    EXPECT_DOUBLE_EQ(gu.f(&X2), 0.078208538795091168);
    // From scipy logpdf(X1,2,25)
    EXPECT_DOUBLE_EQ(gu.lf(&X1), -2.5333764456387726);

    // Initialize the PRNG or set the state of the PRNG
    EXPECT_TRUE(umuq::prng.setState());

    // Produce samples with Exponential distribution density
    gu.sample(&X1);

    // Check for sampling vector
    std::vector<double> Y(15, -1);

    gu.sample(Y, 15);

    // Check for sampling matrix
    umuq::EMatrixX<double> Z(1, 19);

    gu.sample(Z);
}

/*! 
 * \ingroup Test_Module
 * 
 * Test to check multivariateGaussianDistribution 
 */
TEST(densityFunction_test, HandlesMultivariateGaussianDistributionConstruction)
{
    // A multivariate Gaussian distribution with mean zero and unit covariance matrix of size (2*2)
    umuq::multivariateGaussianDistribution<double> m(2);

    // From MATLAB mvnpdf([1.5,2])
    EXPECT_DOUBLE_EQ(m.f(std::vector<double>{1.5, 2}.data()), 0.0069927801704657913);
    // From MATLAB mvnpdf([3,2])
    EXPECT_DOUBLE_EQ(m.f(std::vector<double>{3, 2}.data()), 0.0002392797792004706);

    // Create a covariance matrix
    double M2d[4] = {1, 3. / 5., 3. / 5., 2.};

    // A multivariate Gaussian distribution with mean zero and covariance matrix of M2d
    umuq::multivariateGaussianDistribution<double> mvn(M2d, 2);

    // Prepare data.
    int n = 11;

    // Coordinates
    std::vector<double> x(n * n);
    std::vector<double> x2(n);
    std::vector<double> y(n * n);
    std::vector<double> y2(n);

    // PDF at coordinates
    std::vector<double> pdf(n * n);

    // Log of PDF at coordinates
    std::vector<double> lpdf(n * n);

    {
        double dx = 8. / 10.;

        // Create sample points from Multivariate normal distribution
        for (int i = 0, l = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                x[l] = -4. + dx * i;
                y[l++] = -4. + dx * j;
            }
        }

        for (int i = 0; i < n; i++)
        {
            x2[i] = -4. + dx * i;
        }

        for (int i = 0; i < n; i++)
        {
            y2[i] = -4. + dx * i;
        }
    }

    // Compute PDF at (x,y)
    {
        for (int i = 0; i < n * n; ++i)
        {
            double X[2] = {x[i], y[i]};
            pdf[i] = mvn.f(X);
            lpdf[i] = mvn.lf(X);
        }
    }

#ifdef HAVE_PYTHON
    std::string fileName = "./multivariatescatterpdf.svg";
    std::remove(fileName.c_str());

    // Prepare keywords to pass to PolyCollection. See
    std::map<std::string, std::string> keywords;
    keywords["marker"] = "s";

    // Size
    std::vector<int> s(n * n, 1000);

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x, y, s, pdf, keywords));

    // Add graph title
    EXPECT_TRUE(plt.title("multivariate normal distribution PDF"));

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());

    fileName = "./multivariatecontourpdf.svg";
    std::remove(fileName.c_str());

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Create scatter plot
    EXPECT_TRUE(plt.contourf<double>(x2, y2, pdf));

    // Add graph title
    EXPECT_TRUE(plt.title("multivariate normal distribution PDF contour"));

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());

    fileName = "./multivariatescatterlogpdf.svg";
    std::remove(fileName.c_str());

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x, y, s, lpdf, keywords));

    // Add graph title
    EXPECT_TRUE(plt.title("multivariate normal distribution Log of PDF"));

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());

    fileName = "./multivariatecontourlogpdf.svg";
    std::remove(fileName.c_str());

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Create scatter plot
    EXPECT_TRUE(plt.contourf<double>(x2, y2, lpdf));

    // Add graph title
    EXPECT_TRUE(plt.title("multivariate normal distribution Log of PDF contour"));

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());
#endif

    {
        double X[2];
        // Produce samples with  Multivariate normal distribution density
        mvn.sample(X);
    }

    {
        // Check for sampling vector
        std::vector<double> Y(30, -1);
        mvn.sample(Y, 15);
    }

    {
        // Check for sampling vector
        umuq::EMatrixX<double> Z(2, 19);
        mvn.sample(Z);
    }
}

/*! 
 * \ingroup Test_Module
 * 
 * Test to check multinomialDistribution
 * 
 * Example reference: <br>
 * http://www.probabilityformula.org/multinomial-probability.html
 */
TEST(densityFunction_test, HandlesMultinomialDistributionConstruction)
{
    /*!
     * \brief First example
     *  
     * This is an experiment of drawing a random card from an ordinary playing cards deck is done with replacing it back.
     * This was done ten times. Find the probability of getting 2 spades, 3 diamond, 3 club and 2 hearts.
     */

    {
        // multinomialDistribution distribution where the vector size or types of outputs is 4
        umuq::density::multinomialDistribution<double> m(4);

        // A random sample (with size of K) from the multinomial distribution
        unsigned int X[] = {2, 3, 3, 2};

        // Vector of probabilities \f$ p_1, \cdots, p_k \f$ (with size of K)
        double P[] = {25, 25, 25, 25};

        EXPECT_NEAR(m.f(P, X), 0.024032592773437545, 1e-14);
    }

    /*!
     * \brief Second example
     *  
     * In case of 10 bits, what is the probability that 5 are excellent, 2 are good and 2 are fair and 1 is poor? 
     * Classification of individual bits are independent events and that the probabilities of A, B, C and D are 
     * 40%, 20%, 5% and 1% respectively. 
     * 
     * \note
     * The multinomialDistribution would normalize the probabilities of A, B, C and D to {40/66, 20/66, 5/66, 1/66}.
     */

    {
        // multinomialDistribution distribution where the vector size or types of outputs is 4
        umuq::density::multinomialDistribution<double> m(4);

        // A random sample (with size of K) from the multinomial distribution
        unsigned int X[] = {5, 2, 2, 1};

        // Vector of probabilities \f$ p_1, \cdots, p_k \f$ (with size of K)
        double P[] = {40., 20., 5., 1.};

        EXPECT_NEAR(m.f(P, X), 0.0049360823520927834, 1e-14);
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
