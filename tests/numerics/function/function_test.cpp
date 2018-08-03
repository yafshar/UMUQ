#include "core/core.hpp"
#include "numerics/function/umuqdifferentiablefunction.hpp"
#include "gtest/gtest.h"

/*!
 * \brief Computes the square of x
 * 
 * \param x  Input data
 * \returns  Square of x
 */
double fun_sqrx(double const x)
{
    return x * x;
}

/*!
 * \brief Computes the root square of x
 * 
 * \param x  Input data
 * \returns  Root square of x
 */
double fun_rsqrx(double const x)
{
    return std::sqrt(x);
}

/*! 
 * Test to check function construction
 */
TEST(function_test, HandlesFunctionConstruction)
{
    umuqFunction<double, std::function<double(double const)>> fn("square");

    fn.f = fun_sqrx;
    EXPECT_DOUBLE_EQ(fn.f(2.), 4.0);

    fn.f = fun_rsqrx;
    EXPECT_DOUBLE_EQ(fn.f(4.), 2.0);
}

/*!
 * \brief Differentiable Function test for any general-purpose differentiable function of n variables
 * 
 */

using FUNT = std::function<double(std::vector<double> const &, void *)>;
using DFUNT = std::function<std::vector<double>(std::vector<double> const &, void *)>;

/*!
 * \brief A two-dimensional paraboloid with five parameters
 * 
 * \param v       Input vector of data 
 * \param params  Input parameters
 * 
 * \returns       Function value
 */
double fun_test(std::vector<double> const &v, void *params)
{
    double *p = static_cast<double *>(params);
    return p[2] * (v[0] - p[0]) * (v[0] - p[0]) + p[3] * (v[1] - p[1]) * (v[1] - p[1]) + p[4];
}

//! The gradient of f, df = (df/dx, df/dy)
std::vector<double> dfun_test(std::vector<double> const &v, void *params)
{
    std::vector<double> df(2);
    double *p = static_cast<double *>(params);

    df[0] = 2.0 * p[2] * (v[0] - p[0]);
    df[1] = 2.0 * p[3] * (v[1] - p[1]);

    return df;
}

/*! 
 * Test to check differentiable function construction
 */
TEST(differentiablefunction_test, HandlesDifferentiableFunctionConstruction)
{
    //! create an instance of a differentiable function
    umuqDifferentiableFunction<double, FUNT, DFUNT> fn("paraboloid");

    //! Assigning function and its derivative
    fn.f = fun_test;
    fn.df = dfun_test;

    //! Input parameters
    double p[5] = {1.0, 2.0, 10.0, 20.0, 30.0};
    void *params = static_cast<void *>(p);

    //! Input data point which is the function minimum point
    std::vector<double> x = {1.0, 2.0};
    std::vector<double> dx;
    dx = fn.df(x, params);

    EXPECT_DOUBLE_EQ(fn.f(x, params), 30.0);
    
    EXPECT_DOUBLE_EQ(dx[0], 0.0);
    EXPECT_DOUBLE_EQ(dx[1], 0.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
