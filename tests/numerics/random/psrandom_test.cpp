#include "core/core.hpp"
#include "core/environment.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "io/pyplot.hpp"
#include "gtest/gtest.h"

// Create a global instance of the Pyplot from Pyplot library
pyplot plt;

// Get an instance of a double random object and seed it
psrandom<double> prng(123);

/*! 
 * Test to check random functionality
 */
TEST(random_test, HandlesRandoms)
{
    // Initialize the PRNG or set the state of the PRNG
    EXPECT_TRUE(prng.setState());

    // Create a matrix
    EMatrix2d M2d;
    M2d << 1, 3. / 5., 3. / 5., 2.;

    // Create a zero vector
    EVector2d V2d = EVector2d::Zero();

    // Create an object of type Multivariate normal distribution
    EXPECT_TRUE(prng.set_mvnormal(M2d));

    EVector2d X = prng.mvnormal->dist();

    // Create an object of type Multivariate normal distribution
    EXPECT_TRUE(prng.set_mvnormal(V2d, M2d));

    X = prng.mvnormal->dist();

    // #ifdef HAVE_PYTHON
    //     // Plot line from given x and y data. Color is selected automatically.
    //     EXPECT_TRUE(plt.plot<double>(x, y, "b:", "cos(x)"));
    // #endif

    //! TODO Add the test for checking mvnormdist
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
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}
