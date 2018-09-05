#include "core/core.hpp"
#include "io/io.hpp"
#include "misc/parser.hpp"
#include "numerics/function/fitfunction.hpp"
#include "gtest/gtest.h"

/*!
 * \brief n-dimensional Rosenbrock function
 * 
 * \param x  Input x
 * \param n  dimension of x
 * 
 * \return double 
 */
double f1(double const *x, int const n)
{
    double sum(0);
    for (int i = 0; i < n - 1; i++)
    {
        sum += 100.0 * std::pow((x[i + 1] - x[i] * x[i]), 2) + std::pow((x[i] - 1.0), 2);
    }
    return -std::log(sum);
}

/*! 
 * Test to check fitFunction construction
 * One can easily fix the fitfunction class with any functionality
 */
TEST(fitFunction_test, HandlesfitFunctionConstruction)
{
    fitFunction<double, std::function<double(double const *, int const)>> fit;

    EXPECT_TRUE(fit.setfitFunction(f1));
    EXPECT_TRUE(fit.init());
}

//! Global vector of data
std::vector<double> x;
std::vector<double> y;

/*!
 * \brief Initialization function that we want to use 
 * 
 */
bool init()
{
    umuq::io f;
    {
        // open the data file
        EXPECT_TRUE(f.openFile("./numerics/function/data.txt"));

        //Count number of data points (lines in the file)
        int ndata = 0;
        while (f.readLine())
        {
            ndata++;
        }

        //!Rewind the file
        f.rewindFile();

        x.resize(ndata);
        y.resize(ndata);

        EXPECT_TRUE(f.loadMatrix<double>(x.data(), 1, y.data(), 1, ndata));

        f.closeFile();
    }
    return true;
}

/*!
 * \brief External function to compute the function \f$ y = c_0 \sin(c_1*x+c_2) \f$
 * 
 * \param x Input x coordinates
 * \param y Output computed y \f$ y = c_0 \sin(c_1*x+c_2) \f$
 * \param c model parameters
 */
void f2(std::vector<double> const &x, std::vector<double> &y, double const *c)
{
    auto xit = x.begin();
    auto yit = y.begin();
    for (; xit != x.end(); xit++, yit++)
    {
        *yit = c[0] * std::sin(*xit * c[1] + c[2]);
    }
}

double fitfun(double const *c, int const numc, double *out, int const numout, int const *info)
{
    double sigma2 = std::pow(c[numc - 1], 2);

    int const ndata = x.size();

    std::vector<double> y2(ndata);

    f2(x, y2, c);

    double sum{};
    for (int i = 0; i < ndata; i++)
    {
        sum += std::pow(y[i] - y2[i], 2);
    }

    double res = ndata * M_L2PI + ndata * std::log(sigma2) + sum / sigma2;
    return -res * 0.5;
}

/*!
 * Test to check fitFunction construction with external extension
 */
TEST(fitFunction_test, HandlesexternalfitFunctionConstruction)
{
    //! First we create an instance of the fitFunction object with the default fitting Function type
    fitFunction<double> fit;

    //! We can set the init and fit function together or individually
    EXPECT_TRUE(fit.setInitFunction(init));
    //!
    EXPECT_TRUE(fit.setfitFunction(fitfun));

    //! We call the init function
    EXPECT_TRUE(fit.init());

    //! Check the fit function for two different model parameters coefficient of c1 & c2
    std::vector<double> c1 = {2.0, 3.0, 1.0, 0.5};
    std::vector<double> c2 = {1.0, 1.0, 1.0, 0.5};
    double out[4];

    EXPECT_TRUE(fit.f(c1.data(), 4, out, 4, nullptr) > fit.f(c2.data(), 4, out, 4, nullptr));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
