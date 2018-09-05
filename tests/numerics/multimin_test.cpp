#include "core/core.hpp"
#include "numerics/multimin.hpp"
#include "numerics/testfunctions/optimizationtestfunctions.hpp"
#include "gtest/gtest.h"

/*!
 * \brief A helper function for testing the function minimizer 
 * 
 * \tparam T          Data type
 * \param fMinimizer  Function Minimizer object
 * \param Fun         Function to be minimized 
 * \param X           N-Dimensional input data
 * \param nDim        Dimension of the data
 * \param FunName     Function name or description
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool functionMinimizerTest(umuq::functionMinimizer<T> &fMinimizer, umuq::F_MTYPE<T> const &Fun, T const *X, int const nDim, char const *FunName)
{
    //! By default we consider stepSize 1
    std::vector<T> stepSize(nDim, 1);

    //! First we have to set the minimizer dimension
    if (!fMinimizer.reset(nDim))
    {
        UMUQFAILRETURN("Failed to set the minimizer dimension!");
    }

    //! Second, we have to set the function, input vector and stepsize
    if (!fMinimizer.set(Fun, X, stepSize.data()))
    {
        UMUQFAILRETURN("Failed to set the minimizer!");
    }

    //! Third, initilize the minimizer
    if (!fMinimizer.init())
    {
        UMUQFAILRETURN("Failed to initialize the minimizer!");
    }

#ifdef DEBUG
    {
        T *x = fMinimizer.getX();

        std::cout << "x =";
        for (int i = 0; i < nDim; i++)
        {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
    }
#endif

    //! Forth, iterate until we reach the absolute tolerance of 1e-3

    //! Fail:-1, Success:0, Continue:1
    int status = 1;
    int iter = 0;

    while (iter < 5000 && status == 1)
    {
        iter++;

        if (!fMinimizer.iterate())
        {
            UMUQFAILRETURN("Failed to iterate the minimizer!");
        }

#ifdef DEBUG
        {
            std::cout << iter << ": ";

            T *x = fMinimizer.getX();

            std::cout << "x = ";
            for (int i = 0; i < nDim; i++)
            {
                std::cout << x[i] << " ";
            }
            // std::cout << std::endl;

            std::cout << "f(x) =" << fMinimizer.getMin() << ", & characteristic size =" << fMinimizer.getSize() << std::endl;
        }
#endif

        status = fMinimizer.testSize(1e-3);
    }

    if (status == 0 || status == 1)
    {
        std::cout << fMinimizer.getName() << ", on " << FunName << ": " << iter << " iters, f(x)=" << fMinimizer.getMin() << std::endl;
        std::cout << ((status == 0) ? "Converged to minimum at x = " : "Stopped at x = ");

        T *x = fMinimizer.getX();
        for (int i = 0; i < nDim; i++)
        {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;

        return (status ? (std::abs(fMinimizer.getMin()) > 1e-5) : true);
    }
    return false;
}

/*!
 * \brief A helper function for testing the function minimizer 
 * 
 * \tparam T          Data type
 * 
 * \param fMinimizer  Function Minimizer object
 * \param Fun         Function to be used in this minimizer \f$ f(x) \f$
 * \param DFun        Function gradient $\nabla f$ to be used in this minimizer
 * \param FDFun       Function & its gradient to be used in this minimizer
 * \param X           N-Dimensional input data
 * \param nDim        Dimension of the data
 * \param FunName     Function name or description
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool differentiableFunctionMinimizerTest(umuq::differentiableFunctionMinimizer<T> &fMinimizer,
                                         umuq::F_MTYPE<T> const &Fun, umuq::DF_MTYPE<T> const &DFun, umuq::FDF_MTYPE<T> const &FDFun,
                                         T const *X, int const nDim, char const *FunName)
{
    //! By default we consider stepSize as \f$ 0.1 ||x||_2 = 0.1 \sqrt {\sum x_i^2} \f$
    T stepSize;
    {
        T s(0);
        std::for_each(X, X + nDim, [&](T const x_i) { s += x_i * x_i; });
        stepSize = 0.1 * std::sqrt(s);
    }

    //! First we have to set the minimizer dimension
    if (!fMinimizer.reset(nDim))
    {
        UMUQFAILRETURN("Failed to set the minimizer dimension!");
    }

    //! Second, we have to set the functions (f, df, fdf), input vector, stepsize and tolerance
    if (!fMinimizer.set(Fun, DFun, FDFun, X, stepSize, 0.1))
    {
        UMUQFAILRETURN("Failed to set the minimizer!");
    }

    //! Third, initilize the minimizer
    if (!fMinimizer.init())
    {
        UMUQFAILRETURN("Failed to initialize the minimizer!");
    }

#ifdef DEBUG
    {
        T *x = fMinimizer.getX();

        std::cout << "x =";
        for (int i = 0; i < nDim; i++)
        {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;

        T *g = fMinimizer.getGradient();

        std::cout << "g =";
        for (int i = 0; i < nDim; i++)
        {
            std::cout << g[i] << " ";
        }
        std::cout << std::endl;
    }
#endif

    // Forth, iterate until we reach the absolute tolerance of 1e-3

    // Fail:-1, Success:0, Continue:1
    int status = 1;
    int iter = 0;

    while (iter < 5000 && status == 1)
    {
        iter++;

        if (!fMinimizer.iterate())
        {
            UMUQFAILRETURN("Failed to iterate the minimizer!");
        }

        T *gradient = fMinimizer.getGradient();

#ifdef DEBUG
        {
            std::cout << iter << ": ";

            T *x = fMinimizer.getX();

            std::cout << "x = ";
            for (int i = 0; i < nDim; i++)
            {
                std::cout << x[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "g =";
            for (int i = 0; i < nDim; i++)
            {
                std::cout << gradient[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "f(x) =" << fMinimizer.getMin() << std::endl;

            T *dx = fMinimizer.getdX();

            T s(0);
            std::for_each(dx, dx + nDim, [&](T const d_i) { s += d_i * d_i; });
            std::cout << "dx =" << std::sqrt(s) << std::endl;
        }
#endif

        status = fMinimizer.testGradient(gradient, 1e-3);
    }

    if (status == 0 || status == 1)
    {
        std::cout << fMinimizer.getName() << ", on " << FunName << ": " << iter << " iters, f(x)=" << fMinimizer.getMin() << std::endl;
        std::cout << ((status == 0) ? "Converged to minimum at x = " : "Stopped at x = ");

        T *x = fMinimizer.getX();
        for (int i = 0; i < nDim; i++)
        {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;

        return (status ? (std::abs(fMinimizer.getMin()) > 1e-5) : true);
    }
    return false;
}

/*!
 * \brief A helper function for testing the function minimizer 
 * This is only using the function f as input and df is computed internally
 * 
 * \tparam T          Data type
 * 
 * \param fMinimizer  Function Minimizer object
 * \param Fun         Function to be used in this minimizer \f$ f(x) \f$
 * \param X           N-Dimensional input data
 * \param nDim        Dimension of the data
 * \param FunName     Function name or description
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool differentiableFunctionMinimizerTest(umuq::differentiableFunctionMinimizer<T> &fMinimizer,
                                         umuq::F_MTYPE<T> const &Fun,
                                         T const *X, int const nDim, char const *FunName)
{
    //! By default we consider stepSize as \f$ 0.1 ||x||_2 = 0.1 \sqrt {\sum x_i^2} \f$
    T stepSize;
    {
        T s(0);
        std::for_each(X, X + nDim, [&](T const x_i) { s += x_i * x_i; });
        stepSize = 0.1 * std::sqrt(s);
    }

    //! First we have to set the minimizer dimension
    if (!fMinimizer.reset(nDim))
    {
        UMUQFAILRETURN("Failed to set the minimizer dimension!");
    }

    //! Second, we have to set the functions (f, df, fdf), input vector, stepsize and tolerance
    if (!fMinimizer.set(Fun, X, stepSize, 0.1))
    {
        UMUQFAILRETURN("Failed to set the minimizer!");
    }

    //! Third, initilize the minimizer
    if (!fMinimizer.init())
    {
        UMUQFAILRETURN("Failed to initialize the minimizer!");
    }

#ifdef DEBUG
    {
        T *x = fMinimizer.getX();

        std::cout << "x =";
        for (int i = 0; i < nDim; i++)
        {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;

        T *g = fMinimizer.getGradient();

        std::cout << "g =";
        for (int i = 0; i < nDim; i++)
        {
            std::cout << g[i] << " ";
        }
        std::cout << std::endl;
    }
#endif

    // Forth, iterate until we reach the absolute tolerance of 1e-3

    // Fail:-1, Success:0, Continue:1
    int status = 1;
    int iter = 0;

    while (iter < 5000 && status == 1)
    {
        iter++;

        if (!fMinimizer.iterate())
        {
            UMUQFAILRETURN("Failed to iterate the minimizer!");
        }

        T *gradient = fMinimizer.getGradient();

#ifdef DEBUG
        {
            std::cout << iter << ": ";

            T *x = fMinimizer.getX();

            std::cout << "x = ";
            for (int i = 0; i < nDim; i++)
            {
                std::cout << x[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "g =";
            for (int i = 0; i < nDim; i++)
            {
                std::cout << gradient[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "f(x) =" << fMinimizer.getMin() << std::endl;

            T *dx = fMinimizer.getdX();

            T s(0);
            std::for_each(dx, dx + nDim, [&](T const d_i) { s += d_i * d_i; });
            std::cout << "dx =" << std::sqrt(s) << std::endl;
        }
#endif

        status = fMinimizer.testGradient(gradient, 1e-3);
    }

    if (status == 0 || status == 1)
    {
        std::cout << fMinimizer.getName() << ", on " << FunName << ": " << iter << " iters, f(x)=" << fMinimizer.getMin() << std::endl;
        std::cout << ((status == 0) ? "Converged to minimum at x = " : "Stopped at x = ");

        T *x = fMinimizer.getX();
        for (int i = 0; i < nDim; i++)
        {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;

        return (status ? (std::abs(fMinimizer.getMin()) > 1e-5) : true);
    }
    return false;
}

/*!
 * Test to check multimin functionality if steepestDescent can handle different test functions
 */
TEST(multimin_steepestDescent_test, Handles_TestFunctions)
{
    //! Create an instance of the minimizer
    umuq::steepestDescent<double> fMinimizer;

    //! Starting point for Rosenbrock function
    std::vector<double> X = {-1.2, 1.0};

    //! Check the function minimizer can handle a Rosenbrock function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, rosenbrock_df, rosenbrock_fdf, X.data(), 2, "Rosenbrock"));

    //! Check the function minimizer can handle a Rosenbrock function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, X.data(), 2, "Rosenbrock_F"));

    //! Starting point for Roth function
    X = {4.5, 3.5};

    //! Check the function minimizer can handle a Roth function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, roth_df, roth_fdf, X.data(), 2, "Roth"));

    //! Check the function minimizer can handle a Roth function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, X.data(), 2, "Roth_F"));

    X.resize(4);

    //! Starting point for Wood function
    X = {-3.0, -1, 2.0, 3.0};

    //! Check the function minimizer can handle a Wood function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, wood_df, wood_fdf, X.data(), 4, "Wood"));

    //! Check the function minimizer can handle a Wood function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, X.data(), 4, "Wood_F"));
}

/*!
 * Test to check multimin functionality if conjugatePr can handle different test functions
 */
TEST(multimin_conjugatePr_test, Handles_TestFunctions)
{
    //! Create an instance of the minimizer
    umuq::conjugatePr<double> fMinimizer;

    //! Starting point for Rosenbrock function
    std::vector<double> X = {-1.2, 1.0};

    //! Check the function minimizer can handle a Rosenbrock function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, rosenbrock_df, rosenbrock_fdf, X.data(), 2, "Rosenbrock"));

    //! Check the function minimizer can handle a Rosenbrock function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, X.data(), 2, "Rosenbrock_F"));

    //! Starting point for Roth function
    X = {4.5, 3.5};

    //! Check the function minimizer can handle a Roth function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, roth_df, roth_fdf, X.data(), 2, "Roth"));

    //! Check the function minimizer can handle a Roth function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, X.data(), 2, "Roth_F"));

    X.resize(4);

    //! Starting point for Wood function
    X = {-3.0, -1, 2.0, 3.0};

    //! Check the function minimizer can handle a Wood function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, wood_df, wood_fdf, X.data(), 4, "Wood"));

    //! Check the function minimizer can handle a Wood function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, X.data(), 4, "Wood_F"));
}

/*!
 * Test to check multimin functionality if conjugateFr can handle different test functions
 */
TEST(multimin_conjugateFr_test, Handles_TestFunctions)
{
    //! Create an instance of the minimizer
    umuq::conjugateFr<double> fMinimizer;

    //! Starting point for Rosenbrock function
    std::vector<double> X = {-1.2, 1.0};

    //! Check the function minimizer can handle a Rosenbrock function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, rosenbrock_df, rosenbrock_fdf, X.data(), 2, "Rosenbrock"));

    //! Check the function minimizer can handle a Rosenbrock function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, X.data(), 2, "Rosenbrock_F"));

    //! Starting point for Roth function
    X = {4.5, 3.5};

    //! Check the function minimizer can handle a Roth function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, roth_df, roth_fdf, X.data(), 2, "Roth"));

    //! Check the function minimizer can handle a Roth function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, X.data(), 2, "Roth_F"));

    X.resize(4);

    //! Starting point for Wood function
    X = {-3.0, -1, 2.0, 3.0};

    //! Check the function minimizer can handle a Wood function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, wood_df, wood_fdf, X.data(), 4, "Wood"));

    //! Check the function minimizer can handle a Wood function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, X.data(), 4, "Wood_F"));
}

/*!
 * Test to check multimin functionality if bfgs can handle different test functions
 */
TEST(multimin_bfgs_test, Handles_TestFunctions)
{
    //! Create an instance of the minimizer
    umuq::bfgs<double> fMinimizer;

    //! Starting point for Rosenbrock function
    std::vector<double> X = {-1.2, 1.0};

    //! Check the function minimizer can handle a Rosenbrock function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, rosenbrock_df, rosenbrock_fdf, X.data(), 2, "Rosenbrock"));

    //! Check the function minimizer can handle a Rosenbrock function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, X.data(), 2, "Rosenbrock_F"));

    //! Starting point for Roth function
    X = {4.5, 3.5};

    //! Check the function minimizer can handle a Roth function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, roth_df, roth_fdf, X.data(), 2, "Roth"));

    //! Check the function minimizer can handle a Roth function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, X.data(), 2, "Roth_F"));

    X.resize(4);

    //! Starting point for Wood function
    X = {-3.0, -1, 2.0, 3.0};

    //! Check the function minimizer can handle a Wood function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, wood_df, wood_fdf, X.data(), 4, "Wood"));

    //! Check the function minimizer can handle a Wood function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, X.data(), 4, "Wood_F"));
}

/*!
 * Test to check multimin functionality if bfgs2 can handle different test functions
 */
TEST(multimin_bfgs2_test, Handles_TestFunctions)
{
    //! Create an instance of the minimizer
    umuq::bfgs2<double> fMinimizer;

    //! Starting point for Rosenbrock function
    std::vector<double> X = {-1.2, 1.0};

    //! Check the function minimizer can handle a Rosenbrock function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, rosenbrock_df, rosenbrock_fdf, X.data(), 2, "Rosenbrock"));

    //! Check the function minimizer can handle a Rosenbrock function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, rosenbrock_f, X.data(), 2, "Rosenbrock_F"));

    //! Starting point for Roth function
    X = {4.5, 3.5};

    //! Check the function minimizer can handle a Roth function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, roth_df, roth_fdf, X.data(), 2, "Roth"));

    //! Check the function minimizer can handle a Roth function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, roth_f, X.data(), 2, "Roth_F"));

    X.resize(4);

    //! Starting point for Wood function
    X = {-3.0, -1, 2.0, 3.0};

    //! Check the function minimizer can handle a Wood function using f, df, and fdf
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, wood_df, wood_fdf, X.data(), 4, "Wood"));

    //! Check the function minimizer can handle a Wood function using f
    EXPECT_TRUE(differentiableFunctionMinimizerTest<double>(fMinimizer, wood_f, X.data(), 4, "Wood_F"));
}

/*!
 * Test to check multimin functionality if simplexNM can handle different test functions
 */
TEST(multimin_simplexNM_test, Handles_TestFunctions)
{
    //! Create an instance of the minimizer
    umuq::simplexNM<double> fMinimizer;

    //! Starting point for Rosenbrock function
    std::vector<double> X = {-1.2, 1.0};

    //! Check the function minimizer can handle a Rosenbrock function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, rosenbrock_f, X.data(), 2, "Rosenbrock"));

    //! Starting point for Roth function
    X = {4.5, 3.5};

    //! Check the function minimizer can handle a Roth function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, roth_f, X.data(), 2, "Roth"));

    X.resize(4);

    //! Starting point for Wood function
    X = {-3.0, -1, 2.0, 3.0};

    //! Check the function minimizer can handle a Wood function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, wood_f, X.data(), 4, "Wood"));

    X.resize(3);

    //! Starting point for Spring function
    X = {1.0, 0.0, 7 * M_PI};

    //! Check the function minimizer can handle a Spring function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, spring_f, X.data(), 3, "Spring"));
}

/*!
 * Test to check multimin functionality if simplexNM2 can handle different test functions
 */
TEST(multimin_simplexNM2_test, Handles_TestFunctions)
{
    //! Create an instance of the minimizer
    umuq::simplexNM2<double> fMinimizer;

    //! Starting point for Rosenbrock function
    std::vector<double> X = {-1.2, 1.0};

    //! Check the function minimizer can handle a Rosenbrock function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, rosenbrock_f, X.data(), 2, "Rosenbrock"));

    //! Starting point for Roth function
    X = {4.5, 3.5};

    //! Check the function minimizer can handle a Roth function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, roth_f, X.data(), 2, "Roth"));

    X.resize(4);

    //! Starting point for Wood function
    X = {-3.0, -1, 2.0, 3.0};

    //! Check the function minimizer can handle a Wood function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, wood_f, X.data(), 4, "Wood"));

    X.resize(3);

    //! Starting point for Spring function
    X = {1.0, 0.0, 7 * M_PI};

    //! Check the function minimizer can handle a Spring function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, spring_f, X.data(), 3, "Spring"));
}

/*!
 * Test to check multimin functionality if simplexNM2Rnd can handle different test functions
 */
TEST(multimin_simplexNM2Rnd_test, Handles_TestFunctions)
{
    //! Create an instance of the minimizer
    umuq::simplexNM2Rnd<double> fMinimizer;

    //! Starting point for Rosenbrock function
    std::vector<double> X = {-1.2, 1.0};

    //! Check the function minimizer can handle a Rosenbrock function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, rosenbrock_f, X.data(), 2, "Rosenbrock"));

    //! Starting point for Roth function
    X = {4.5, 3.5};

    //! Check the function minimizer can handle a Roth function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, roth_f, X.data(), 2, "Roth"));

    X.resize(4);

    //! Starting point for Wood function
    X = {-3.0, -1, 2.0, 3.0};

    //! Check the function minimizer can handle a Wood function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, wood_f, X.data(), 4, "Wood"));

    X.resize(3);

    //! Starting point for Spring function
    X = {1.0, 0.0, 7 * M_PI};

    //! Check the function minimizer can handle a Spring function using f
    EXPECT_TRUE(functionMinimizerTest<double>(fMinimizer, spring_f, X.data(), 3, "Spring"));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
