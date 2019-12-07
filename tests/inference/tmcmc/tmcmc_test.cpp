#include "inference/tmcmc/tmcmc.hpp"
#include "environment.hpp"
#include "gtest/gtest.h"

// Global vector of data
std::vector<double> x;
std::vector<double> y;

/*!
 * \ingroup Test_Module
 *
 * \brief Initialization function that we want to use
 *
 */
bool init()
{
    umuq::io f;
    {
        // open the data file
        EXPECT_TRUE(f.openFile("./function/data.txt"));

        //Count number of data points (lines in the file)
        int ndata = 0;
        while (f.readLine())
        {
            ndata++;
        }

        //Rewind the file
        f.rewindFile();

        x.resize(ndata);
        y.resize(ndata);

        EXPECT_TRUE(f.loadMatrix(x.data(), 1, y.data(), 1, ndata));

        f.closeFile();
    }
    return true;
}

/*!
 * \ingroup Test_Module
 *
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

/*!
 * \ingroup Test_Module
 *
 * \brief Testing fitting function
 *
 * \param c      Input parameter
 * \param numc   Number of input parameters
 * \param out    Output
 * \param numout Number of outputs
 * \param info   Task information
 *
 * \returns log likelihood
 */
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

// Tests tmcmc
TEST(tmcmc_test, HandlesConstruction)
{
    // Create an instance of the tmcmc object
    umuq::tmcmc::tmcmc<> t;

    // Set the input file
    EXPECT_TRUE(t.setInputFileName("./tmcmc/test.txt"));

    std::cout << t.inputFilename << std::endl;

    // Initializing the object before setting the fitting function is wrong!
    EXPECT_FALSE(t.init());

    // reset the TMCMC object based on the read data
    EXPECT_TRUE(t.reset());

    // // Set the init and fit function together or individually
    // EXPECT_TRUE(t.setFitFunction(init, fitfun));

    // // Initializing the object before setting the fitting function is wrong!
    // EXPECT_TRUE(t.init());
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}
