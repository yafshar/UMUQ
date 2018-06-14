#include "core/core.hpp"
#include "core/environment.hpp"
#include "numerics/eigenmatrix.hpp"
#include "numerics/psrandom.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check random functionality
 */
TEST(random_test, HandlesRandoms)
{
    //get an instance of a random object and seed it
    psrandom r(123);
    EXPECT_TRUE(r.init());

    //create a matrix
    EMatrix2d EM2d;
    EM2d << 1, 3. / 5., 3. / 5., 2.;

    //create a zero vector
    EVector2d EV2d = EVector2d::Zero();

    //create an object of type Multivariate normal distribution
    mvnormdist<double> MVNOBJ1(EM2d);

    EVector2d X = MVNOBJ1();

    MVNOBJ1.pdf(X);
    MVNOBJ1.lnpdf(X);

    //create an object of type Multivariate normal distribution
    Mvnormdist<double> MVNOBJ2(EV2d, EM2d);

    X = MVNOBJ2();
    MVNOBJ2.pdf(X);
    MVNOBJ2.lnpdf(X);

    //TODO Add the test for checking mvnormdist
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}
