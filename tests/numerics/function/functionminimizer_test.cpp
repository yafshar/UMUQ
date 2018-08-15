#include "core/core.hpp"
#include "numerics/function/umuqdifferentiablefunction.hpp"
#include "numerics/function/functionminimizer.hpp"
#include "gtest/gtest.h"

/*!
 * \brief Computes \f$ x^2 \times y^3 \f$
 * 
 * \param x  Input data
 * \param y  Input data
 * 
 * \returns  \f$ x^2 \times y^3 \f$
 */
double f1(double const x, double const y)
{
    return std::pow(x, 2) * std::pow(y, 3);
}

/*!
 * \brief Computes df = (df/dx, df/dy)
 * 
 * \param x  Input data
 * \param y  Input data
 * 
 * \returns df
 */
double *f2(double const x, double const y)
{
    double *df = new double[2];
    df[0] = 2 * x * std::pow(y, 3);
    df[1] = std::pow(x, 2) * 3 * std::pow(y, 2);
    return df;
}

/*!
 * \brief Computes \f$ f=x \times y, \text{ and } df = (df/dx, df/dy) \f$
 * 
 * \param x  Input data
 * \param y  Input data
 * \param f  Output function value 
 * \param df Output function value 
 * 
 * \returns  f & df
 */
void f3(double const *x, double const *y, double *f, double *df)
{
    double X = *x;
    double Y = *y;

    if (f)
    {
        *f = std::pow(X, 2) * std::pow(Y, 3);
    }

    if (df)
    {
        df[0] = 2 * X * std::pow(Y, 3);
        df[1] = std::pow(X, 2) * 3 * std::pow(Y, 2);
    }
}

double f4(double const x, double const y)
{
    return std::pow(x, 4) * std::pow(y, 3);
}

double *f5(double const x, double const y)
{
    double *df = new double[2];
    df[0] = 4 * std::pow(x, 3) * std::pow(y, 3);
    df[1] = std::pow(x, 4) * 3 * std::pow(y, 2);
    return df;
}

void f6(double const *x, double const *y, double *f, double *df)
{
    double X = *x;
    double Y = *y;

    if (f)
    {
        *f = std::pow(X, 4) * std::pow(Y, 3);
    }

    if (df)
    {
        df[0] = 4 * std::pow(X, 3) * std::pow(Y, 3);
        df[1] = std::pow(X, 4) * 3 * std::pow(Y, 2);
    }
}

/*! 
 * Test to check functionminimizer construction
 */
TEST(function_test, HandlesFunctionMinimizerConstruction)
{
    umuqFunction<double, std::function<double(double const, double const)>> fn("product");
    fn.f = f1;

    std::vector<double> x(2, 10.);
    std::vector<double> s(2, 0.1);

    functionMinimizer<double, std::function<double(double const, double const)>> fnm("fmin");
    fnm.reset(2);
    EXPECT_TRUE(fnm.set(fn, x, s));

    EXPECT_DOUBLE_EQ(fnm.fun.f(2., 3), 108.);

    //! Set the target function to a new function
    EXPECT_TRUE(fnm.set(f4, x, s));

    EXPECT_DOUBLE_EQ(fnm.fun.f(2., 3), 432.);
}

using FT = std::function<double(double const, double const)>;
using DFT = std::function<double *(double const, double const)>;
using FDFT = std::function<void(double const *, double const *, double *, double *)>;

/*! 
 * Test to check differentiablefunctionminimizer construction
 */
TEST(differentiablefunction_test, HandlesDifferentiableFunctionMinimizerConstruction)
{
    umuqDifferentiableFunction<double, FT, DFT, FDFT> fn("product2");
    fn.f = f1;
    fn.df = f2;
    fn.fdf = f3;

    std::vector<double> x(2, 10.);
    std::vector<double> s(2, 0.1);
    double tol = 0.1;

    differentiableFunctionMinimizer<double, FT, DFT, FDFT> fnm("fmin");

    //! First we have to set the dimension, otherwise we can not set the function
    EXPECT_FALSE(fnm.set(fn, x, s, tol));

    //! Set the dimension
    fnm.reset(2);

    //! After setting the dimesnion, now we can set the function
    EXPECT_TRUE(fnm.set(fn, x, s, tol));

    EXPECT_DOUBLE_EQ(fnm.fun.f(2., 3), 108.);

    //! Set the target function to new functions
    EXPECT_TRUE(fnm.set(f4, f5, f6, x, s, tol));

    //! Check to make sure it is set to the correct function
    EXPECT_DOUBLE_EQ(fnm.fun.f(2., 3), 432.);

    //! Create a new minimizer
    differentiableFunctionMinimizer<double, FT, DFT, FDFT> fnm2("fmin2");
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
