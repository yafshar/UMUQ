#include "core/core.hpp"
#include "numerics/function/functionminimizer.hpp"
#include "numerics/multimin/simplexnm2.hpp"
#include "numerics/testfunctions/optimizationtestfunctions.hpp"
#include "gtest/gtest.h"

/*! 
 * \ingroup Test_Module
 * 
 * \brief Test to check simplexNM2 construction
 */
TEST(simplexNM2_test, HandlesMinimizerConstruction)
{
    umuq::simplexNM2<double> fMinimizer;

    //! First we have to set the minimizer dimension
    EXPECT_TRUE(fMinimizer.reset(2));
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Test if simplexNM2 can handle a Rosenbrock function
 * 
 */
TEST(simplexNM2_test, HandlesRosenbrockFunction)
{
    //! Starting point
    std::vector<double> X = {-1.2, 1.0};

    //! By default we consider stepSize 1
    std::vector<double> stepSize(2, 1);

    //! Create an instance of the minimizer
    umuq::simplexNM2<double> fMinimizer;

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

/*!
 * \ingroup Test_Module
 * 
 * \brief Test if simplexNM2 can handle a Roth function
 * 
 */
TEST(simplexNM2_test, HandlesRothFunction)
{
    //! Starting point
    std::vector<double> X = {4.5, 3.5};

    //! By default we consider stepSize 1
    std::vector<double> stepSize(2, 1);

    //! Create an instance of the minimizer
    umuq::simplexNM2<double> fMinimizer;

    //! First we have to set the minimizer dimension
    EXPECT_TRUE(fMinimizer.reset(2));

    //! Second, we have to set the function, input vector, stepsize and tolerance
    EXPECT_TRUE(fMinimizer.set(roth_f, X, stepSize));

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

    std::cout << fMinimizer.getName() << ", on Roth function : " << iter << " iters, f(x)=" << fMinimizer.getMin() << std::endl;
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
