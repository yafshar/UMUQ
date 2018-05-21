#include "core/core.hpp"
#include "io/io.hpp"
#include "numerics/dcpse.hpp"
#include "numerics/testfunctions/predictiontestfunctions.hpp"
#include "gtest/gtest.h"

template <typename T>
void fillPagebyPage(T *idata, T *coords, int const d, T lx, T ly, T dx, T dy, int x, int y)
{
    for (int r = 0; r < x; r++)
    {
        for (int c = 0; c < y; c++)
        {
            std::copy(coords, coords + d, idata);
            idata += d;
            *idata++ = lx + r * dx;
            *idata++ = ly + c * dy;
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
    if (nDim < 1)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Wrong dimension ! " << std::endl;
        return false;
    }

    //compute total number of points
    int nPoints(1);
    std::for_each(nDPoints, nDPoints + nDim, [&](int const d_i) { nPoints *= d_i; });

    if (idata == nullptr)
    {
        try
        {
            idata = new T[nPoints * nDim];
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
        for (int i = 0; i < nPoints; i++)
        {
            idata[i] = Lb[0] + i * dx[0];
        }
        break;
    case 2:
    {
        for (int i = 0, n = 0; i < nDPoints[0]; i++)
        {
            T const r = Lb[0] + i * dx[0];
            for (int j = 0; j < nDPoints[1]; j++)
            {
                idata[n] = r;
                idata[n + 1] = Lb[1] + j * dx[1];
                n += 2;
            }
        }
    }
    break;
    default:
        // int nd1 = nDim - 1;
        // int nd2 = nDim - 2;
        // int nd3 = nDim - 3;

        // T *d;

        // T *coords = new T[nd2];

        // std::ptrdiff_t counter = 0;

        // for (int i = 0; i < std::accumulate(nDPoints, nDPoints + nd2, 1, std::multiplies<int>()); ++i)
        // {
        //     for (int j = nd3; j >= 0; --j)
        //     {
        //         int const fId = std::floor(counter / std::accumulate(nDPoints + j + 1, nDPoints + nDim, nDim, std::multiplies<int>()));
        //         coords[j] = Lb[j] + dx[j] * (fId >= nDPoints[j] ? fId % nDPoints[j] : fId);
        //         std::cout << "counter=" << counter << " i=" << i << " ,j=" << j << " dx=" << dx[j] << " coords=" << coords[j] << std::endl;
        //     }

        //     d = idata + counter;

        //     fillPagebyPage<T>(d, coords, nd2, Lb[nd2], Lb[nd1], dx[nd2], dx[nd1], nDPoints[nd2], nDPoints[nd1]);

        //     counter += nDim * nDPoints[nd2] * nDPoints[nd1];
        // }

        // delete[] coords;
        break;
    }

    delete[] dx;
    return true;
}

/*! \fn meshgrid
 * \brief Compute N-D coordinates randomly distributed between \f$ [Lb \cdots Ub] \f$
 * 
 * \tparam T        data type
 * 
 * \param idata     N-D coordinates 
 * \param nPoints   Total number of points
 * \param nDim      Dimensionality
 * \param Lb        Lower bound in each dimension
 * \param Ub        Upper bound in each dimension
 * 
 * \return true     if it successfully creates N-D grid coordinates
 */
template <typename T>
bool meshgrid(T *&idata, int const nPoints, int const nDim, double *Lb, double *Ub)
{
    if (nDim < 1)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Wrong dimension ! " << std::endl;
        return false;
    }

    if (idata == nullptr)
    {
        try
        {
            idata = new T[nPoints * nDim];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }
    }

    // std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    switch (nDim)
    {
    case 1:
        for (int i = 0; i < nPoints; i++)
        {
            idata[i] = Lb[0] + dis(gen) * (Ub[0] - Lb[0]);
        }
        break;

    case 2:
        for (int i = 0, n = 0; i < nPoints; i++)
        {
            idata[n] = Lb[0] + dis(gen) * (Ub[0] - Lb[0]);
            idata[n + 1] = Lb[1] + dis(gen) * (Ub[1] - Lb[1]);
            n += 2;
        }
        break;

    default:
        for (int i = 0, n = 0; i < nPoints; i++)
        {
            for (int d = 0; d < nDim; d++, n++)
            {
                idata[n] = Lb[d] + dis(gen) * (Ub[d] - Lb[d]);
            }
        }
        break;
    }

    return true;
}

/*! 
 * Test to check dcpse functionality for frank2d function
 */
TEST(dcpse_test, HandlesFrank2dFunctionCartesianPoints)
{
    int nDim = 2;
    int nPoints = 41;
    int nqpoints = 20;

    double *idata = nullptr;

    int nDPoints[] = {nPoints, nPoints};
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
        iFvalue = new double[nPoints * nPoints];
        qFvalue = new double[nqpoints];
        qFvalueExact = new double[nqpoints];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    //create an instance of frank2d object
    franke2d<double> f2d;

    double *data = idata;
    for (int i = 0; i < nPoints * nPoints; i++)
    {
        iFvalue[i] = f2d.f(data);
        data += 2;
    }

    {
        // std::random_device rd;
        std::mt19937 gen(1);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0, l = 0; i < nqpoints; i++)
        {
            qdata[l] = dis(gen);
            qdata[l + 1] = dis(gen);
            l += 2;
        }

        for (int i = 0, n = 0; i < nqpoints; i++)
        {
            qFvalueExact[i] = f2d.f(qdata + n);
            n += nDim;
        }
    }

    //Create an instance of a DC-PSE object
    dcpse<double> dc(nDim);

    //Compute the interpolator weights
    EXPECT_TRUE(dc.computeInterpolatorWeights(idata, nPoints * nPoints, qdata, nqpoints));

    //Compute the operator kernel for interpolation
    EXPECT_TRUE(dc.interpolate(iFvalue, nPoints * nPoints, qFvalue, nqpoints));

    // TEMapVectorX<double> A(qFvalueExact, nqpoints);
    // TEMapVectorX<double> B(qFvalue, nqpoints);

    // std::cout << "Relative error = " << (A-B).norm() << std::endl;

    // //Create an instance of io object
    // io file;

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_EXACT", file.in | file.out | file.trunc))
    // {
    //     for (int i = 0, n = 0; i < nPoints * nPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(idata + n, 1, nDim, 2);
    //         file.saveMatrix<double>(iFvalue + i, 1, 1);
    //         n += nDim;
    //     }
    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_DCPSE", file.in | file.out | file.trunc))
    // {
    //     for (int i = 0, n = 0; i < nqpoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata + n, 1, nDim, 2);
    //         file.saveMatrix<double>(qFvalue + i, 1, 1);
    //         n += nDim;
    //     }

    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_DCPSE_EXACT", file.in | file.out | file.trunc))
    // {
    //     for (int i = 0, n = 0; i < nqpoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata + n, 1, nDim, 2);
    //         file.saveMatrix<double>(qFvalueExact + i, 1, 1);
    //         n += nDim;
    //     }

    //     file.closeFile();
    // }

    delete[] qFvalueExact;
    delete[] qFvalue;
    delete[] iFvalue;
    delete[] qdata;
    delete[] idata;
}

/*! 
 * Test to check dcpse functionality for frank2d function
 */
TEST(dcpse_test, HandlesFrank2dFunctionRandomPoints)
{
    int nDim = 2;
    int nPoints = 41;
    int nqpoints = 20;

    double *idata = nullptr;

    double Lb[] = {0, 0};
    double Ub[] = {1, 1};

    EXPECT_TRUE(meshgrid<double>(idata, nPoints * nPoints, nDim, Lb, Ub));

    double *qdata = nullptr;
    double *iFvalue = nullptr;
    double *qFvalue = nullptr;
    double *qFvalueExact = nullptr;

    try
    {
        qdata = new double[nqpoints * nDim];
        iFvalue = new double[nPoints * nPoints];
        qFvalue = new double[nqpoints];
        qFvalueExact = new double[nqpoints];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    //create an instance of frank2d object
    franke2d<double> f2d;

    for (int i = 0, n = 0; i < nPoints * nPoints; i++)
    {
        iFvalue[i] = f2d.f(idata + n);
        n += nDim;
    }

    {
        // std::random_device rd;
        std::mt19937 gen(1);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0, l = 0; i < nqpoints; i++)
        {
            qdata[l] = dis(gen);
            qdata[l + 1] = dis(gen);
            l += 2;
        }

        for (int i = 0, n = 0; i < nqpoints; i++)
        {
            qFvalueExact[i] = f2d.f(qdata + n);
            n += nDim;
        }
    }

    //Create an instance of a DC-PSE object
    dcpse<double> dc(nDim);

    //Compute the interpolator weights
    EXPECT_TRUE(dc.computeInterpolatorWeights(idata, nPoints * nPoints, qdata, nqpoints));

    //Compute the operator kernel for interpolation
    EXPECT_TRUE(dc.interpolate(iFvalue, nPoints * nPoints, qFvalue, nqpoints));

    // TEMapVectorX<double> A(qFvalueExact, nqpoints);
    // TEMapVectorX<double> B(qFvalue, nqpoints);

    // std::cout << "Relative error = " << (A-B).norm() << std::endl;

    // //Create an instance of io object
    // io file;

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_EXACTRND", file.in | file.out | file.trunc))
    // {
    //     for (int i = 0, n = 0; i < nPoints * nPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(idata + n, 1, nDim, 2);
    //         file.saveMatrix<double>(iFvalue + i, 1, 1);
    //         n += nDim;
    //     }
    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_DCPSERND", file.in | file.out | file.trunc))
    // {
    //     for (int i = 0, n = 0; i < nqpoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata + n, 1, nDim, 2);
    //         file.saveMatrix<double>(qFvalue + i, 1, 1);
    //         n += nDim;
    //     }

    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_DCPSE_EXACTRND", file.in | file.out | file.trunc))
    // {
    //     for (int i = 0, n = 0; i < nqpoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata + n, 1, nDim, 2);
    //         file.saveMatrix<double>(qFvalueExact + i, 1, 1);
    //         n += nDim;
    //     }

    //     file.closeFile();
    // }

    delete[] qFvalueExact;
    delete[] qFvalue;
    delete[] iFvalue;
    delete[] qdata;
    delete[] idata;
}

/*! 
 * Test to check dcpse functionality for Rastrigin function
 */
TEST(dcpse_test, HandlesRastriginFunctionCartesianPoints)
{
    int nDim = 2;
    int nPoints = 41;
    int nqpoints = 20;

    double *idata = nullptr;

    int nDPoints[] = {nPoints, nPoints};
    double Lb[] = {-5.12, -5.12};
    double Ub[] = {5.12, 5.12};

    EXPECT_TRUE(meshgrid<double>(idata, nDPoints, nDim, Lb, Ub));

    double *qdata = nullptr;
    double *iFvalue = nullptr;
    double *qFvalue = nullptr;
    double *qFvalueExact = nullptr;

    try
    {
        qdata = new double[nqpoints * nDim];
        iFvalue = new double[nPoints * nPoints];
        qFvalue = new double[nqpoints];
        qFvalueExact = new double[nqpoints];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    //create an instance of rastrigin 2d object
    rastrigin<double> r2d(nDim);

    for (int i = 0, n = 0; i < nPoints * nPoints; i++)
    {
        iFvalue[i] = r2d.f(idata + n);
        n += 2;
    }

    {
        // std::random_device rd;
        std::mt19937 gen(1);
        std::uniform_real_distribution<> dis(-5.12, 5.12);

        for (int i = 0, l = 0; i < nqpoints; i++)
        {
            qdata[l] = dis(gen);
            qdata[l + 1] = dis(gen);
            l += 2;
        }

        for (int i = 0, n = 0; i < nqpoints; i++)
        {
            qFvalueExact[i] = r2d.f(qdata + n);
            n += nDim;
        }
    }

    //Create an instance of a DC-PSE object
    dcpse<double> dc(nDim);

    //Compute the interpolator weights
    EXPECT_TRUE(dc.computeInterpolatorWeights(idata, nPoints * nPoints, qdata, nqpoints));

    //Compute the operator kernel for interpolation
    EXPECT_TRUE(dc.interpolate(iFvalue, nPoints * nPoints, qFvalue, nqpoints));

    // TEMapVectorX<double> A(qFvalueExact, nqpoints);
    // TEMapVectorX<double> B(qFvalue, nqpoints);

    // std::cout << "Relative error = " << (A-B).norm() << std::endl;

    // //Create an instance of io object
    // io file;

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/RASTRIGIN2D_EXACT", file.in | file.out | file.trunc))
    // {
    //     for (int i = 0, n = 0; i < nPoints * nPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(idata + n, 1, nDim, 2);
    //         file.saveMatrix<double>(iFvalue + i, 1, 1);
    //         n += nDim;
    //     }
    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/RASTRIGIN2D_DCPSE", file.in | file.out | file.trunc))
    // {
    //     for (int i = 0, n = 0; i < nqpoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata + n, 1, nDim, 2);
    //         file.saveMatrix<double>(qFvalue + i, 1, 1);
    //         n += nDim;
    //     }

    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/RASTRIGIN2D_DCPSE_EXACT", file.in | file.out | file.trunc))
    // {
    //     for (int i = 0, n = 0; i < nqpoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata + n, 1, nDim, 2);
    //         file.saveMatrix<double>(qFvalueExact + i, 1, 1);
    //         n += nDim;
    //     }

    //     file.closeFile();
    // }

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
