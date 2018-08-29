#include "core/core.hpp"
#include "core/environment.hpp"
#include "numerics/factorial.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/function/densityfunction.hpp"
#include "numerics/density/uniformdistribution.hpp"
#include "numerics/density/exponentialdistribution.hpp"
#include "numerics/density/gammadistribution.hpp"
#include "numerics/density/gaussiandistribution.hpp"
#include "numerics/density/multivariategaussiandistribution.hpp"
#include "io/pyplot.hpp"
#include "gtest/gtest.h"

// Create a global instance of the Pyplot from Pyplot library
pyplot plt;

/*! 
 * Test to check densityFunction functionality
 */
TEST(densityFunction_test, HandlesConstruction)
{
    std::cout << std::fixed;

    //! Uniform ditrsibution between 1 and 2
    uniformDistribution<double> u(1, 2);
    double X1 = 1.5;
    double X2 = 3.;
    EXPECT_DOUBLE_EQ(u.f(&X1), 1.);
    EXPECT_DOUBLE_EQ(u.f(&X2), 0.);

    //! Exponential distribution with mean 1
    exponentialDistribution<float> e(1);
    float X3 = 1.5f;
    float X4 = 3.f;

    EXPECT_FLOAT_EQ(e.f(&X3), std::exp(-X3));
    EXPECT_FLOAT_EQ(e.f(&X4), std::exp(-X4));

    //! Gamma distribution with Shape parameter of 0.5
    gammaDistribution<double> g(0.5);

    //! From MATLAB gampdf(X1, 0.5, 1)
    EXPECT_DOUBLE_EQ(g.f(&X1), 0.10278688653584618); 
    //! From MATLAB gampdf(X2, 0.5, 1)
    EXPECT_DOUBLE_EQ(g.f(&X2), 0.01621739110988048);
                               
    //! Gaussian distribution with mean 2 and standard deviation of 5
    gaussianDistribution<double> gu(2, 5);
    //! From MATLAB normpdf(X1,2,5)
    EXPECT_DOUBLE_EQ(gu.f(&X1), 0.079390509495402356);
    //! From MATLAB normpdf(X2, 2, 5)
    EXPECT_DOUBLE_EQ(gu.f(&X2), 0.078208538795091168);

    //! A multivariate Gaussian distribution with mean zero and unit covariance matrix of size (2*2)
    multivariateGaussianDistribution<double> m(2);
    //! From MATLAB mvnpdf([1.5,2])
    EXPECT_DOUBLE_EQ(m.f(std::vector<double>{1.5, 2}.data()), 0.0069927801704657913);
    //! From MATLAB mvnpdf([3,2])
    EXPECT_DOUBLE_EQ(m.f(std::vector<double>{3, 2}.data()), 0.0002392797792004706);


    //! Create a covariance matrix
    double M2d[4] = {1, 3. / 5., 3. / 5., 2.};

    //! A multivariate Gaussian distribution with mean zero and covariance matrix of M2d
    multivariateGaussianDistribution<double> mvn(M2d, 2);

    //! Prepare data.
    int n = 11;

    //! X coordinates
    std::vector<double> x(n * n);
    std::vector<double> y(n * n);

    //! PDF at coordinates
    std::vector<double> pdf(n * n);

    //! Log of PDF at coordinates
    std::vector<double> lpdf(n * n);

    {
        double dx = 8. / 10.;

        //! Create sample points from Multivariate normal distribution
        for (int i = 0, l = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                x[l] = -4. + dx * i;
                y[l++] = -4. + dx * j;
            }
        }
    }

    //! Compute PDF at (x,y)
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

    //! Prepare keywords to pass to PolyCollection. See
    std::map<std::string, std::string> keywords;
    keywords["marker"] = "o";

    //! Size
    std::vector<double> s(n * n, 16);

    //! Clear previous plot
    EXPECT_TRUE(plt.clf());

    //! Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x, y, s, pdf, keywords));

    //! Add graph title
    EXPECT_TRUE(plt.title("multivariate normal distribution PDF"));

    //! save figure
    EXPECT_TRUE(plt.savefig(fileName));

    fileName = "./multivariatescatterlogpdf.svg";
    std::remove(fileName.c_str());

    //! Clear previous plot
    EXPECT_TRUE(plt.clf());

    //! Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x, y, s, lpdf, keywords));

    //! Add graph title
    EXPECT_TRUE(plt.title("multivariate normal distribution Log of PDF"));

    //! save figure
    EXPECT_TRUE(plt.savefig(fileName));
#endif
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new UMUQEventListener);

    return RUN_ALL_TESTS();
}