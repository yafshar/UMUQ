#include "core/core.hpp"
#include "io/io.hpp"
#include "numerics/dcpse.hpp"
#include "gtest/gtest.h"

template <typename T>
void fillPagebyPage(T *idata, T *coords, int const d, T dx, T dy, int x, int y)
{
    for (int r = 0; r < x; r++)
    {
        for (int c = 0; c < y; c++)
        {
            std::copy(coords, coords + d, idata);
            idata += d;
            *idata++ = r * dx;
            *idata++ = c * dy;
        }
    }
}

/*! \fn meshgrid
 * \brief Compute N-D grid coordinates between \f$ [Lb \cdots Ub] \f$
 * 
 * \tparam T        data type
 * 
 * \param idata     N-D grid coordinates 
 * \param nDPoints  Number of points in each direction
 * \param nDim      Dimensionality
 * \param Lb        Lower bound in each dimension
 * \param Ub        Upper bound in each dimension
 * 
 * \return true     if it successfully creates N-D grid coordinates
 */
template <typename T>
bool meshgrid(T *&idata, int const *nDPoints, int const nDim, double *Lb, double *Ub)
{
    //compute total number of points
    int npoints(1);
    std::for_each(nDPoints, nDPoints + nDim, [&](int const d_i) { npoints *= d_i; });

    if (idata == nullptr)
    {
        try
        {
            idata = new T[npoints * nDim];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }
    }

    double *dx = nullptr;
    try
    {
        dx = new double[nDim];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        return false;
    }
    {
        double *d = dx;
        double *u = Ub;
        double *l = Lb;
        std::for_each(nDPoints, nDPoints + nDim, [&](int const d_i) { *d++ = (*u++ - *l++) / (d_i - 1); });
    }

    switch (nDim)
    {
    case 1:
        for (int i = 0; i < npoints; i++)
        {
            idata[i] = i * dx[0];
        }
        break;
    case 2:
    {
        for (int i = 0, n = 0; i < nDPoints[0]; i++)
        {
            T const r = i * dx[0];
            for (int j = 0; j < nDPoints[1]; j++)
            {
                idata[n] = r;
                idata[n + 1] = j * dx[1];
                n += 2;
            }
        }
    }
    break;
        // default:
        //     int nd1 = nDim - 1;
        //     int nd2 = nDim - 2;
        //     int nd3 = nDim - 3;

        //     T *d;

        //     T *coords = new T[nd2];

        //     std::ptrdiff_t counter = 0;

        //     for (int i = 0; i < nDim; ++i)
        //     {
        //         for (int j = nd3; j >= 0; --j)
        //         {
        //             int const fId = std::floor(counter / std::accumulate(nDPoints + j + 1, nDPoints + nDim, nDim, std::multiplies<int>()));
        //             coords[j] = dx[j] * (fId >= nDPoints[j] ? fId % nDPoints[j] : fId);
        //             std::cout << "counter=" << counter << " i=" << i << " ,j=" << j << " dx=" << dx[j] << " coords=" << coords[j] << std::endl;
        //         }

        //         d = idata + counter;

        //         fillPagebyPage<T>(d, coords, nd2, dx[nd2], dx[nd1], nDPoints[nd2], nDPoints[nd1]);

        //         counter += nDim * nDPoints[nd2] * nDPoints[nd1];
        //     }

        //     delete[] coords;
        //     break;
    }

    delete[] dx;
    return true;
}

/*!
 * \brief Franke's bivariate test function
 * 
 * Franke's bivariate test function is a weighted sum of four exponentials
 * \f[
 * f(x) &= 0.75 e^\left(-\frac{(9x_1-2)^2}{4} - \frac{(9x_2-2)^2}{4} \right) \\
 *      &+ 0.75 e^\left(-\frac{(9x_1+1)^2}{49} - \frac{(9x_2+1)}{10} \right) \\
 *      &+ 0.5 e^\left(-\frac{(9x_1-7)^2}{4} - \frac{(9x_2-3)^2}{4} \right) \\
 *      &- 0.2 e^\left(-(9x_1-4)^2 - (9x_2-7)^2 \right)
 * \f]
 * 
 * \tparam T    data type
 * \param idata input data point
 * \return T    function value at input data point
 */
template <typename T>
inline T franke2d(T const *idata)
{
    T const x1 = idata[0];
    T const x2 = idata[1];
    T const t1 = 0.75 * std::exp(-std::pow(9 * x1 - 2, 2) / 4 - std::pow(9 * x2 - 2, 2) / 4);
    T const t2 = 0.75 * std::exp(-std::pow(9 * x1 + 1, 2) / 49 - (9 * x2 + 1) / 10);
    T const t3 = 0.5 * std::exp(-std::pow(9 * x1 - 7, 2) / 4 - std::pow(9 * x2 - 3, 2) / 4);
    T const t4 = -0.2 * std::exp(-std::pow(9 * x1 - 4, 2) - std::pow(9 * x2 - 7, 2));
    return t1 + t2 + t3 + t4;
}

/*! 
 * Test to check dcpse functionality
 */
TEST(dcpse_test, HandlesDCoperators)
{
    int nDim = 2;
    int npoints = 21;
    int nqpoints = 4;

    double *idata = nullptr;

    int nDPoints[] = {npoints, npoints};
    double Lb[] = {0, 0};
    double Ub[] = {1, 1};

    EXPECT_TRUE(meshgrid<double>(idata, nDPoints, nDim, Lb, Ub));

    double *qdata = nullptr;
    double *iFvalue = nullptr;
    double *qFvalue = nullptr;
    double *qFvalueExact = nullptr;

    try
    {
        qdata = new double[nqpoints * nDim];
        iFvalue = new double[npoints * npoints];
        qFvalue = new double[nqpoints];
        qFvalueExact = new double[nqpoints];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    double *data = idata;
    for (int i = 0; i < npoints * npoints; i++)
    {
        iFvalue[i] = franke2d<double>(data);
        data += 2;
    }

    {
        std::random_device rd;
        std::mt19937 gen(1);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0, l = 0; i < nqpoints; i++)
        {
            qdata[l] = dis(gen);
            qdata[l + 1] = dis(gen);
            l += 2;
        }

        std::swap(qdata[0], qdata[nqpoints * nDim - 2]);
        std::swap(qdata[1], qdata[nqpoints * nDim - 1]);

        for (int i = 0, n = 0; i < nqpoints; i++)
        {
            qFvalueExact[i] = franke2d<double>(qdata + n);
            n += nDim;
        }
    }

    dcpse<double> dc(nDim);

    EXPECT_TRUE(dc.computeInterpolatorWeights(idata, npoints * npoints, qdata, nqpoints));

    EXPECT_TRUE(dc.interpolate(iFvalue, npoints * npoints, qFvalue, nqpoints));

    io f;

    //!Open a file for reading and writing
    if (f.openFile("Exact", f.in | f.out | f.trunc))
    {
        for (int i = 0, n = 0; i < npoints * npoints; i++)
        {
            //!Write the matrix in it
            f.saveMatrix<double>(idata + n, 1, nDim, 2);
            f.saveMatrix<double>(iFvalue + i, 1, 1);
            n += nDim;
        }

        f.closeFile();
    }

    //!Open a file for reading and writing
    if (f.openFile("DCPSE", f.in | f.out | f.trunc))
    {
        for (int i = 0, n = 0; i < nqpoints; i++)
        {
            //!Write the matrix in it
            f.saveMatrix<double>(qdata + n, 1, nDim, 2);
            f.saveMatrix<double>(qFvalue + i, 1, 1);
            n += nDim;
        }

        f.closeFile();
    }

    //!Open a file for reading and writing
    if (f.openFile("QExact", f.in | f.out | f.trunc))
    {
        for (int i = 0, n = 0; i < nqpoints; i++)
        {
            //!Write the matrix in it
            f.saveMatrix<double>(qdata + n, 1, nDim, 2);
            f.saveMatrix<double>(qFvalueExact + i, 1, 1);
            n += nDim;
        }

        f.closeFile();
    }

    delete[] qFvalueExact;
    delete[] qFvalue;
    delete[] iFvalue;
    delete[] qdata;
    delete[] idata;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
