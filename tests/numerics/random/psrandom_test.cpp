#include "core/core.hpp"
#include "environment.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "io/io.hpp"
#include "io/pyplot.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 * 
 * \brief  Create a global instance of the Pyplot from Pyplot library
 * 
 */
umuq::pyplot plt;

/*!
 * \ingroup Test_Module
 * 
 * \brief Get an instance of a seeded double random object
 * 
 */
umuq::psrandom prng(123);

/*! 
 * \ingroup Test_Module
 * 
 * Test to check random functionality
 */
TEST(random_test, HandlesRandoms)
{
    // Initialize the PRNG or set the state of the PRNG
    EXPECT_TRUE(prng.setState());

    // Create a matrix
    umuq::EMatrix2d M2d;
    M2d << 1, 3. / 5., 3. / 5., 2.;

    // Create a zero vector
    umuq::EVector2d V2d = umuq::EVector2d::Zero();

    // Create an object of type Multivariate normal distribution
    umuq::randomdist::multivariateNormalDistribution<double> mvnormal(M2d);

    umuq::EVector2d X = mvnormal.dist();

    // Create an object of type Multivariate normal distribution
    mvnormal = std::move(umuq::randomdist::multivariateNormalDistribution<double>(V2d, M2d));

    X = mvnormal.dist();

#ifdef HAVE_PYTHON
    std::string fileName = "./multivariatescatterpoints.svg";
    std::remove(fileName.c_str());

    //! Prepare data.
    int n = 500;

    //! X coordinates
    std::vector<double> x(n);
    //! Y coordinates
    std::vector<double> y(n);

    //! Create sample points from Multivariate normal distribution
    for (int i = 0; i < n; ++i)
    {
        X = mvnormal.dist();
        x[i] = X[0];
        y[i] = X[1];
    }
    //! Prepare keywords to pass to PolyCollection. See
    std::map<std::string, std::string> keywords;
    keywords["marker"] = "s";

    //! Clear previous plot
    EXPECT_TRUE(plt.clf());

    //! Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x, y, 800, "lime", keywords));

    //! Add graph title
    EXPECT_TRUE(plt.title("Sample points from a multivariate normal distribution"));

    //! save figure
    EXPECT_TRUE(plt.savefig(fileName));

    //! close figure
    EXPECT_TRUE(plt.close());
#endif
}

/*! 
 * \ingroup Test_Module
 * 
 * Test to check random functionality
 */
TEST(random_test, HandlesMultivariate)
{
    //     EMatrixXd idata(3, 2);
    //     idata << 4.348817, 2.995049, -3.793431, 4.711934, 1.190864, -1.357363;

    // cov(samples) # 19.03539 11.91384 \n 11.91384  9.28796

    // Initialize the PRNG or set the state of the PRNG
    EXPECT_TRUE(prng.setState());

    std::vector<double> Mean{3., 2.};
    std::vector<double> Covariance{10., 5., 5., 5.};

    std::vector<double> a(2);

    // Map the data to the Eigen vector format
    umuq::EVectorMapType<double> Ea(a.data(), 2);

    // Create an object of type Multivariate normal distribution
    umuq::randomdist::multivariateNormalDistribution<double> mvnormal(Mean.data(), Covariance.data(), 2);

    Ea = mvnormal.dist();
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
