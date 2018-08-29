#include "core/core.hpp"
#include "io/io.hpp"
#include "misc/parser.hpp"
#include "numerics/function/fitfunction.hpp"
#include "gtest/gtest.h"

using f1TYPE = std::function<double(double const *, int const)>;

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
    fitFunction<double, f1TYPE> fitrosenbrock;

    EXPECT_TRUE(fitrosenbrock.set(f1));
}

/*!
 * \brief Extend the fitfunction class for any use 
 * 
 */
class fitTest : public fitFunction<double>
{
  public:
    bool init()
    {
        // open the data file
        std::string filename = "./numerics/function/data.txt";

        io f;
        {
            EXPECT_TRUE(f.openFile(filename));
            int ndata = 0;
            while (f.readLine())
            {
                ndata++;
            }

            //!Rewind the file
            f.rewindFile();

            this->x.resize(ndata * 2);

            EXPECT_TRUE(f.loadMatrix<double>(this->x.data(), ndata, 2));

            f.closeFile();
        }
        return true;
    }
};

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
    for (; xit != x.end(); xit += 2, yit++)
    {
        *yit = c[0] * std::sin(c[1] * *xit + c[2]);
    }
}

double fitfun(long long const other, double const *c, int const n, double *out, int const *info)
{
    // To use the information from other class
    auto obj = reinterpret_cast<fitFunction<double> *>(other);

    double sigma2 = std::pow(c[n - 1], 2);

    int const N = obj->x.size();

    std::vector<double> y(N);

    f2(obj->x, y, c);

    double sum{};
    for (int i = 0, j = 1; i < N; i++, j += 2)
    {
        sum += std::pow(obj->x[j] - y[i], 2);
    }

    double res = N * M_L2PI + N * std::log(sigma2) + sum / sigma2;
    return -res * 0.5;
}

/*!
 * Test to check fitFunction construction with external extension
 */
TEST(fitFunction_test, HandlesexternalfitFunctionConstruction)
{
    fitTest fitExternal;
    EXPECT_TRUE(fitExternal.init());
    EXPECT_TRUE(fitExternal.set(fitfun));

    std::vector<double> c = {2., 3.0, 1.0, 0.5};
    double out[4];
    std::cout << fitExternal.fun.f(reinterpret_cast<long long>(&fitExternal), c.data(), 4, out, nullptr) << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
