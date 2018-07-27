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
    std::cout << std::fixed;

    //! Uniform ditrsibution between 1 and 2
    uniformDistribution<double> u(1, 2);
    std::cout << "uniformDistribution =              " << std::cout.width(32) << u.f(1.5) << "  " << std::cout.width(32) << u.f(3) << std::endl;

    EXPECT_DOUBLE_EQ(u.f(3), double{});

    //! Exponential distribution with mean 1
    exponentialDistribution<float> e(1);
    std::cout << "exponentialDistribution =          " << std::cout.width(32) << e.f(1.5) << "  " << std::cout.width(32) << e.f(3) << std::endl;

    //! Gamma distribution with Shape parameter of 0.5
    gammaDistribution<double> g(0.5);
    std::cout << "gammaDistribution =                " << std::cout.width(32) << g.f(1.5) << "  " << std::cout.width(20) << g.f(3) << std::endl;

    //! Gaussian distribution with mean 2 and standard deviation of 5
    gaussianDistribution<double> gu(2, 5);
    std::cout << "gaussianDistribution =             " << std::cout.width(20) << gu.f(1.5) << "  " << std::cout.width(20) << gu.f(3) << std::endl;

    //! A multivariate Gaussian distribution with mean zero and unit covariance matrix of size (2*2)
    multivariateGaussianDistribution<double> m(2);
    std::cout << "multivariateGaussianDistribution = " << std::cout.width(20) << m.f(std::vector<double>{1.5, 2}.data()) << "  " << std::cout.width(20) << m.f(std::vector<double>{3, 2}.data()) << std::endl;

    //! Create a covariance matrix
    double M2d[4] ={1, 3. / 5., 3. / 5., 2.};

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

    return RUN_ALL_TESTS();
}