#include "torc.h"
#include "core/core.hpp"
#include "numerics/eigenmatrix.hpp"
#include "numerics/psrandom.hpp"
#include "gtest/gtest.h"

class TORCEnvironment : public ::testing::Environment
{
  public:
    virtual void SetUp()
    {
        char **argv;
        int argc = 0;

        torc_init(argc, argv, 0);

        // ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
        // if (torc_node_id() != 0)
        // {
        //     delete listeners.Release(listeners.default_result_printer());
        // }
    }

    virtual void TearDown()
    {
        torc_finalize();
    }

    virtual ~TORCEnvironment() {}
};

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

    //create an object of type Multivariate normal distribution
    Mvnormdist<double> MVNOBJ2(EV2d, EM2d);

    //TODO Add the test for mvnormdist

    std::cout << MVNOBJ2() << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new TORCEnvironment);

    return RUN_ALL_TESTS();
}
