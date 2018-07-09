#include "core/core.hpp"
#include "core/environment.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "io/pyplot.hpp"
#include "gtest/gtest.h"
// Create a global instance of the Pyplot from Pyplot library
pyplot plt;

/*! 
 * Test to check random functionality
 */
TEST(random_test, HandlesRandoms)
{
	typedef EMatrix2<double> EMatrix2d;
	typedef EVector2<double> EVector2d;

    // Get an instance of a double random object and seed it
    psrandom<double> r(123);

    // Initialize the PRNG or set the state of the PRNG
    EXPECT_TRUE(r.setState());

    // Create a matrix
	EMatrix2d M2d;
    M2d << 1, 3. / 5., 3. / 5., 2.;

    // Create a zero vector
	EVector2d V2d = EVector2d::Zero();

    // Create an object of type Multivariate normal distribution
    EXPECT_TRUE(r.set_mvnormal(M2d));

    EVector2d X = r.mvnormal->dist();

    // Create an object of type Multivariate normal distribution
    EXPECT_TRUE(r.set_mvnormal(V2d, M2d));

    X = r.mvnormal->dist();

// #ifdef HAVE_PYTHON
//     // Plot line from given x and y data. Color is selected automatically.
//     EXPECT_TRUE(plt.plot<double>(x, y, "b:", "cos(x)"));
// #endif 

    //! TODO Add the test for checking mvnormdist
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}
