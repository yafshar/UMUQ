#include "numerics/random/psrandom.hpp"
#include "datatype/eigendatatype.hpp"
#include "numerics/eigenlib.hpp"
#include "io/pyplot.hpp"
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
 * \brief An instance of the Pyplot from Pyplot library
 *
 */
umuq::matplotlib_223::pyplot plt;

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

    EXPECT_DOUBLE_EQ((V2d - mvnormal.mean).norm(), 0.0);

    // Create an object of type Multivariate normal distribution
    mvnormal = std::move(umuq::randomdist::multivariateNormalDistribution<double>(V2d, M2d));

    EXPECT_DOUBLE_EQ((V2d - mvnormal.mean).norm(), 0.0);
    EXPECT_DOUBLE_EQ((M2d - mvnormal.covariance).norm(), 0.0);

    umuq::EVector2d X = mvnormal.dist();

    {
        umuq::EVector2d Y = mvnormal.dist();

        EXPECT_TRUE(((X - Y).norm() > 0));
    }

#ifdef HAVE_PYTHON
    std::string fileName = "./multivariatescatterpoints.png";
    std::remove(fileName.c_str());

    //! Prepare data.
    int n = 500;

    //! X coordinates
    std::vector<double> x1(n), x2(n);
    //! Y coordinates
    std::vector<double> y1(n), y2(n);

    //! Create sample points from Multivariate normal distribution
    for (int i = 0; i < n; ++i)
    {
        X = mvnormal.dist();
        x1[i] = X[0];
        y1[i] = X[1];
        X = mvnormal.dist();
        x2[i] = X[0];
        y2[i] = X[1];
    }
    //! Prepare keywords to pass to PolyCollection. See
    std::map<std::string, std::string> keywords;

    //! Clear previous plot
    EXPECT_TRUE(plt.clf());
    // '8': 'octagon'
    keywords["marker"] = "8";
    //! Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x1, y1, 100, "lime", keywords));
    // "^": 'triangle_up'
    keywords["marker"] = "^";
    //! Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x2, y2, 80, "b", keywords));

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

    {
        umuq::EVector2d Y = mvnormal.dist();

        EXPECT_TRUE(((Ea - Y).norm() > 0));
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
