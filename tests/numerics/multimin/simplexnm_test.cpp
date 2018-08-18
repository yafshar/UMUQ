#include "core/core.hpp"
#include "numerics/multimin/simplexnm.hpp"
#include "numerics/testfunctions/optimizationtestfunctions.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check simplexNM construction
 */
TEST(simplexNM_test, HandlesMinimizerConstruction)
{
    //! Create an instance of the minimizer
    simplexNM<double> fMinimizer;

    //! First we have to set the minimizer dimension
    EXPECT_TRUE(fMinimizer.reset(2));
}

/*!
 * \brief Test if simplexNM can handle a Rosenbrock function
 * 
 */
TEST(simplexNM_test, HandlesRosenbrockFunction)
{
    //! Starting point
    std::vector<double> X = {-1.2, 1.0};

    //! By default we consider stepSize 1
    std::vector<double> stepSize(2, 1);

    //! Create an instance of the minimizer
    simplexNM<double> fMinimizer;

    //! First we have to set the minimizer dimension
    EXPECT_TRUE(fMinimizer.reset(2));

    //! Second, we have to set the function, input vector and stepsize
    EXPECT_TRUE(fMinimizer.set(rosenbrock_f, X, stepSize));

    //! Third, initilize the minimizer
    EXPECT_TRUE(fMinimizer.init());

    // Forth, iterate until we reach the absolute tolerance of 1e-3

    // Fail:-1, Success:0, Continue:1
    int status = 1;
    int iter = 0;
    while (iter < 5000 && status == 1)
    {
        iter++;
        EXPECT_TRUE(fMinimizer.iterate());
        status = fMinimizer.testSize(1e-3);
    }

    //! Check to see if we succeeded
    EXPECT_TRUE(status == 0);

    std::cout << fMinimizer.getName() << ", on Rosenbrock function : " << iter << " iters, f(x)=" << fMinimizer.getMin() << std::endl;
    std::cout << "Converged to minimum at x=";

    double *x = fMinimizer.getX();
    for (int i = 0; i < 2; i++)
    {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
