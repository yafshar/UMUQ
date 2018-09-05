#include "core/core.hpp"
#include "environment.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "io/pyplot.hpp"
#include "gtest/gtest.h"

// Create a global instance of the Pyplot from Pyplot library
umuq::pyplot plt;

// Get an instance of a double random object and seed it
umuq::psrandom<double> prng(123);

/*! 
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
    EXPECT_TRUE(prng.set_mvnormal(M2d));

    umuq::EVector2d X = prng.mvnormal->dist();

    // Create an object of type Multivariate normal distribution
    EXPECT_TRUE(prng.set_mvnormal(V2d, M2d));

    X = prng.mvnormal->dist();

#ifdef HAVE_PYTHON
    std::string fileName = "./multivariatescatterpoints.svg";
    std::remove(fileName.c_str());

    //! Prepare data.
    int n = 100;

    //! X coordinates
    std::vector<double> x(n);
    //! Y coordinates
    std::vector<double> y(n);

    //! Create sample points from Multivariate normal distribution
    for (int i = 0; i < n; ++i)
    {
        X = prng.mvnormal->dist();
        x[i] = X[0];
        y[i] = X[1];
    }
    //! Prepare keywords to pass to PolyCollection. See
    std::map<std::string, std::string> keywords;
    keywords["marker"] = "D";

    //! Clear previous plot
    EXPECT_TRUE(plt.clf());

    //! Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x, y, 20, 2, keywords));

    //! Add graph title
    EXPECT_TRUE(plt.title("Sample points from a multivariate normal distribution"));

    //! save figure
    EXPECT_TRUE(plt.savefig(fileName));
#endif
}

/*! 
 * Test to check random functionality
 */
TEST(random_test, HandlesMultivariate)
{
    //     EMatrixXd idata(3, 2);
    //     idata << 4.348817, 2.995049, -3.793431, 4.711934, 1.190864, -1.357363;

    // cov(samples) # 19.03539 11.91384 \n 11.91384  9.28796
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
