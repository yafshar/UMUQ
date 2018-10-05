#include "core/core.hpp"
#include "numerics/function/umuqdifferentiablefunction.hpp"
#include "numerics/function/functionminimizer.hpp"
#include "numerics/function/differentiablefunctionminimizer.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 * 
 * \brief Computes \f$ x_0^2 \times x_1^3 \f$
 * 
 * \param x  Input data
 * 
 * \returns  \f$ x_0^2 \times x_1^3 \f$
 */
double f1(double const *x)
{
    return std::pow(x[0], 2) * std::pow(x[1], 3);
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Computes df = (df/dx, df/dy)
 * 
 * \param x   Input data
 * \param df  Output gradient
 * 
 * \return true 
 * \return false 
 */
bool f2(double const *x, double *df)
{
    if (df)
    {
        df[0] = 2 * x[0] * std::pow(x[1], 3);
        df[1] = std::pow(x[0], 2) * 3 * std::pow(x[1], 2);
        return true;
    }
    UMUQFAILRETURN("The gradient pointer is not assigned!");
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Computes \f$ f=x_0 \times x_1, \text{ and } df = (df/dx_0, df/dx_1) \f$
 * 
 * \param x  Input data
 * \param f  Output function value 
 * \param df Output function value 
 * 
 * \return true 
 * \return false 
 */
bool f3(double const *x, double *f, double *df)
{
    if (f)
    {
        *f = std::pow(x[0], 2) * std::pow(x[1], 3);
    }
    else
    {
        UMUQFAILRETURN("The function pointer is not assigned!");
    }

    if (df)
    {
        df[0] = 2 * x[0] * std::pow(x[1], 3);
        df[1] = std::pow(x[0], 2) * 3 * std::pow(x[1], 2);
        return true;
    }
    UMUQFAILRETURN("The gradient pointer is not assigned!");
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Computes \f$ x_0^4 \times x_1^3 \f$
 * 
 * \param x  Input data
 * 
 * \returns  \f$ x_0^4 \times x_1^3 \f$
 */
double f4(double const *x)
{
    return std::pow(x[0], 4) * std::pow(x[1], 3);
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Computes df = (df/dx, df/dy)
 * 
 * \param x   Input data
 * \param df  Output gradient
 * 
 * \return true 
 * \return false 
 */
bool f5(double const *x, double *df)
{
    if (df)
    {
        df[0] = 4 * std::pow(x[0], 3) * std::pow(x[1], 3);
        df[1] = std::pow(x[0], 4) * 3 * std::pow(x[1], 2);
        return true;
    }
    UMUQFAILRETURN("The gradient pointer is not assigned!");
}

/*!
 * \ingroup Test_Module
 * 
 * \brief Computes \f$ f=x_0 \times x_1, \text{ and } df = (df/dx_0, df/dx_1) \f$
 * 
 * \param x  Input data
 * \param f  Output function value 
 * \param df Output function value 
 * 
 * \return true 
 * \return false 
 */
bool f6(double const *x, double *f, double *df)
{
    if (f)
    {
        *f = std::pow(x[0], 2) * std::pow(x[1], 3);
    }
    else
    {
        UMUQFAILRETURN("The function pointer is not assigned!");
    }

    if (df)
    {
        df[0] = 4 * std::pow(x[0], 3) * std::pow(x[1], 3);
        df[1] = std::pow(x[0], 4) * 3 * std::pow(x[1], 2);
        return true;
    }
    UMUQFAILRETURN("The gradient pointer is not assigned!");
}

/*! 
 * \ingroup Test_Module
 * 
 * Test to check functionminimizer construction
 */
TEST(function_test, HandlesFunctionMinimizerConstruction)
{
    umuq::umuqFunction<double, umuq::F_MTYPE<double>> fn("product");
    fn.f = f1;

    std::vector<double> xi(2);

    //! Creating the starting point and step size vector
    std::vector<double> x(2, 10.);
    std::vector<double> s(2, 0.1);

    umuq::functionMinimizer<double> fnm("fmin");

    //! First we have to set the dimension
    fnm.reset(2);

    //! Second we need to assign the function for this minimizer
    EXPECT_TRUE(fnm.set(fn, x, s));

    xi = {2.0, 3.0};

    //! Check if the assigned function works correctly
    EXPECT_DOUBLE_EQ(fnm.fun.f(xi.data()), 108.);

    //! Set the target function to a new function
    EXPECT_TRUE(fnm.set(f4, x, s));

    //! Check if the new assigned function works correctly
    EXPECT_DOUBLE_EQ(fnm.fun.f(xi.data()), 432.);
}

/*! 
 * \ingroup Test_Module
 * 
 * Test to check differentiablefunctionminimizer construction
 */
TEST(differentiablefunction_test, HandlesDifferentiableFunctionMinimizerConstruction)
{
    umuq::umuqDifferentiableFunction<double, umuq::F_MTYPE<double>, umuq::DF_MTYPE<double>, umuq::FDF_MTYPE<double>> fn("product2");
    fn.f = f1;
    fn.df = f2;
    fn.fdf = f3;

    std::vector<double> xi(2);

    //! Creating the starting point and step size vector
    std::vector<double> x(2, 10.);

    double s = 0.1;
    
    //! For differentiable minimizer, we need to also set the tolerance
    double tol = 0.1;

    //! Create an instance of the minimizer
    umuq::differentiableFunctionMinimizer<double> fnm("fmin");

    //! First we have to set the dimension, otherwise we can not set the function
    EXPECT_FALSE(fnm.set(fn, x, s, tol));

    //! Set the dimension
    fnm.reset(2);

    //! After setting the dimesnion, now we can set the function
    EXPECT_TRUE(fnm.set(fn, x, s, tol));

    //! Check if the assigned function works correctly
    xi = {2.0, 3.0};
    EXPECT_DOUBLE_EQ(fnm.fun.f(xi.data()), 108.);

    //! Set the target function to a new functions
    EXPECT_TRUE(fnm.set(f4, f5, f6, x, s, tol));

    //! Check to make sure it is set to the correct function
    EXPECT_DOUBLE_EQ(fnm.fun.f(xi.data()), 432.);

    //! Create a new minimizer
    umuq::differentiableFunctionMinimizer<double> fnm2("fmin2");
    fnm2.reset(2);

    //! Set the minimizer functions
    fnm2.fun.f = f1;
    fnm2.fun.df = f2;
    fnm2.fun.fdf = f3;

    //! set the input vector
    EXPECT_TRUE(fnm2.set(x, s, tol));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
