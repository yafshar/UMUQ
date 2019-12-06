#include "numerics/function/umuqdifferentiablefunction.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *
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
 * \ingroup Test_Module
 *
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
 * \ingroup Test_Module
 *
 * Test to check function construction
 */
TEST(function_test, HandlesFunctionConstruction)
{
    umuq::umuqFunction<double, std::function<double(double const)>> fn("square");

    fn.f = fun_sqrx;
    EXPECT_TRUE(fn);
    EXPECT_DOUBLE_EQ(fn.f(2.), 4.0);

    fn.f = fun_rsqrx;
    EXPECT_DOUBLE_EQ(fn.f(4.), 2.0);
}

/*!
 * \ingroup Test_Module
 *
 * \brief Differentiable Function test for any general-purpose differentiable function of n variables
 *
 */

using FUNT = std::function<double(std::vector<double> const &, void *)>;
using DFUNT = std::function<std::vector<double>(std::vector<double> const &, void *)>;

/*!
 * \ingroup Test_Module
 *
 * \brief A two-dimensional paraboloid with five parameters
 *
 * \param v       Input vector of data
 * \param params  Input parameters
 *
 * \returns       Function value
 */
double f_test(std::vector<double> const &v, void *params)
{
    double *p = static_cast<double *>(params);
    return p[2] * (v[0] - p[0]) * (v[0] - p[0]) + p[3] * (v[1] - p[1]) * (v[1] - p[1]) + p[4];
}

/*!
 * \ingroup Test_Module
 *
 * \brief The gradient of f, df = (df/dx, df/dy)
 *
 * \param v
 * \param params
 * \returns std::vector<double>
 */
std::vector<double> df_test(std::vector<double> const &v, void *params)
{
    //Creating output vector
    std::vector<double> df(2);

    double *p = static_cast<double *>(params);

    df[0] = 2.0 * p[2] * (v[0] - p[0]);
    df[1] = 2.0 * p[3] * (v[1] - p[1]);

    return df;
}

/*!
 * \ingroup Test_Module
 *
 * \brief Function and its derivative
 *
 * \param v
 * \param p
 * \param f
 * \param df
 */
void fdf_test(double const *v, double const *p, double *f, double *df)
{
    *f = p[2] * (v[0] - p[0]) * (v[0] - p[0]) + p[3] * (v[1] - p[1]) * (v[1] - p[1]) + p[4];
    df[0] = 2.0 * p[2] * (v[0] - p[0]);
    df[1] = 2.0 * p[3] * (v[1] - p[1]);
}

/*!
 * \ingroup Test_Module
 *
 * Test to check differentiable function construction
 */
TEST(differentiablefunction_test, HandlesDifferentiableFunctionConstruction)
{
    //! Input parameters
    double p[5] = {1.0, 2.0, 10.0, 20.0, 30.0};

    //! create an instance of a differentiable function
    umuq::umuqDifferentiableFunction<double, FUNT, DFUNT> fn(p, 5, "paraboloid");

    //! Assigning function and its derivative
    fn.f = f_test;
    fn.df = df_test;
    fn.fdf = fdf_test;

    EXPECT_TRUE(fn);

    void *params = static_cast<void *>(p);

    //! Input data point which is the function minimum point
    std::vector<double> x = {1.0, 2.0};
    std::vector<double> dx;
    dx = fn.df(x, params);

    EXPECT_DOUBLE_EQ(fn.f(x, params), 30.0);

    EXPECT_DOUBLE_EQ(dx[0], 0.0);
    EXPECT_DOUBLE_EQ(dx[1], 0.0);

    double F;
    double DF[2];

    fn.fdf(x.data(), fn.params.data(), &F, DF);

    EXPECT_DOUBLE_EQ(F, 30.0);
    EXPECT_DOUBLE_EQ(DF[0], 0.0);
    EXPECT_DOUBLE_EQ(DF[1], 0.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
