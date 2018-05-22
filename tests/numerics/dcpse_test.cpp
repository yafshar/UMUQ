#include "core/core.hpp"
#include "io/io.hpp"
#include "numerics/dcpse.hpp"
#include "numerics/testfunctions/predictiontestfunctions.hpp"
#include "gtest/gtest.h"

template <typename T>
void fillPagebyPage(T *idata, T *coords, int const d, T lx, T ly, T dx, T dy, int x, int y)
{
    T *data = idata;
    for (int r = 0; r < x; r++)
    {
        for (int c = 0; c < y; c++)
        {
            std::copy(coords, coords + d, data);
            data += d;
            *data++ = lx + r * dx;
            *data++ = ly + c * dy;
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
 * Test to check dcpse functionality for Qian function
 */
TEST(dcpse_test, HandlesQianFunction)
{
    int nDim = 1;
    double Lb[] = {0};
    double Ub[] = {1};
    int nDPoints[] = {20};
    int nPoints = std::accumulate(nDPoints, nDPoints + nDim, 1, std::multiplies<int>());
    int nqPoints = 3;

    std::unique_ptr<double[]> idata;
    std::unique_ptr<double[]> iqdata;
    std::unique_ptr<double[]> iFvalue;
    std::unique_ptr<double[]> iqFvalue;
    std::unique_ptr<double[]> iqFvalueExact;

    double *data;
    double *qdata;
    double *fvalue;
    double *qfvalue;

    try
    {
        idata.reset(new double[nPoints * nDim]);
        iqdata.reset(new double[nqPoints * nDim]);

        iFvalue.reset(new double[nPoints]);
        iqFvalue.reset(new double[nqPoints]);
        iqFvalueExact.reset(new double[nqPoints]);
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    //Create input points
    data = idata.get();
    EXPECT_TRUE(meshgrid<double>(data, nDPoints, nDim, Lb, Ub));

    //Create an instance of frank2d object
    qian<double> Qian;

    //Compute the function value at each input point
    {
        data = idata.get();
        for (int i = 0; i < nPoints; i++)
        {
            iFvalue[i] = Qian.f(data);
            data += nDim;
        }
    }

    //Create random query data points
    {
        qdata = iqdata.get();

        // std::random_device rd;
        std::mt19937 gen(1);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < nqPoints; i++)
        {
            std::for_each(qdata, qdata + nDim, [&](double &i) { i = dis(gen); });
            qdata += nDim;
        }

        //Compute the function value at each query point
        qdata = iqdata.get();
        for (int i = 0; i < nqPoints; i++)
        {
            iqFvalueExact[i] = Qian.f(qdata);
            qdata += nDim;
        }
    }

    //Create an instance of a DC-PSE object
    dcpse<double> dc(nDim);

    data = idata.get();
    qdata = iqdata.get();

    //Compute the interpolator weights & operator kernel
    EXPECT_TRUE(dc.computeInterpolatorWeights(data, nPoints, qdata, nqPoints, 3));

    fvalue = iFvalue.get();
    qfvalue = iqFvalue.get();

    //Compute the interpolated values
    EXPECT_TRUE(dc.interpolate(fvalue, nPoints, qfvalue, nqPoints));

    {
        int order = dc.orderofAccuracy();

        std::cout << "DC-PSE uses " << dc.neighborhoodKernelSize() << " number of points in the neighborhood of each query points to do a \n"
                  << order << (order == 1 ? "st  " : order == 2 ? "nd  " : order == 3 ? "rd  " : "th  ") << "order interpolation." << std::endl;
    }

    // //Create an instance of io object
    // io file;

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/QIAN_EXACT", file.in | file.out | file.trunc))
    // {
    //     data = idata.get();
    //     fvalue = iFvalue.get();
    //     for (int i = 0; i < nPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(data, 1, nDim, 2);
    //         file.saveMatrix<double>(fvalue, 1, 1);
    //         data += nDim;
    //         fvalue++;
    //     }
    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/QIAN_DCPSE", file.in | file.out | file.trunc))
    // {
    //     qdata = iqdata.get();
    //     qfvalue = iqFvalue.get();
    //     for (int i = 0; i < nqPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata, 1, nDim, 2);
    //         file.saveMatrix<double>(qfvalue, 1, 1);
    //         qdata += nDim;
    //         qfvalue++;
    //     }

    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/QIAN_DCPSE_EXACT", file.in | file.out | file.trunc))
    // {
    //     qdata = iqdata.get();
    //     qfvalue = iqFvalueExact.get();
    //     for (int i = 0; i < nqPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata, 1, nDim, 2);
    //         file.saveMatrix<double>(qfvalue, 1, 1);
    //         qdata += nDim;
    //         qfvalue++;
    //     }

    //     file.closeFile();
    // }
}

/*! 
 * Test to check dcpse functionality for frank2d function
 */
TEST(dcpse_test, HandlesFrank2dFunctionCartesianPoints)
{
    int nDim = 2;
    double Lb[] = {0, 0};
    double Ub[] = {1, 1};
    int nDPoints[] = {21, 21};
    int nPoints = std::accumulate(nDPoints, nDPoints + nDim, 1, std::multiplies<int>());
    int nqPoints = 20;

    std::unique_ptr<double[]> idata;
    std::unique_ptr<double[]> iqdata;
    std::unique_ptr<double[]> iFvalue;
    std::unique_ptr<double[]> iqFvalue;
    std::unique_ptr<double[]> iqFvalueExact;

    double *data;
    double *qdata;
    double *fvalue;
    double *qfvalue;

    try
    {
        idata.reset(new double[nPoints * nDim]);
        iqdata.reset(new double[nqPoints * nDim]);

        iFvalue.reset(new double[nPoints]);
        iqFvalue.reset(new double[nqPoints]);
        iqFvalueExact.reset(new double[nqPoints]);
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    //Create input points
    data = idata.get();
    EXPECT_TRUE(meshgrid<double>(data, nDPoints, nDim, Lb, Ub));

    //create an instance of frank2d object
    franke2d<double> f2d;

    //Compute the function value at each input point
    {
        data = idata.get();
        for (int i = 0; i < nPoints; i++)
        {
            iFvalue[i] = f2d.f(data);
            data += nDim;
        }
    }

    //Create random query data points
    {
        qdata = iqdata.get();

        // std::random_device rd;
        std::mt19937 gen(1);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < nqPoints; i++)
        {
            std::for_each(qdata, qdata + nDim, [&](double &i) { i = dis(gen); });
            qdata += nDim;
        }

        //Compute the function value at each query point
        qdata = iqdata.get();
        for (int i = 0; i < nqPoints; i++)
        {
            iqFvalueExact[i] = f2d.f(qdata);
            qdata += nDim;
        }
    }

    //Create an instance of a DC-PSE object
    dcpse<double> dc(nDim);

    data = idata.get();
    qdata = iqdata.get();

    //Compute the interpolator weights & operator kernel
    EXPECT_TRUE(dc.computeInterpolatorWeights(data, nPoints, qdata, nqPoints));

    fvalue = iFvalue.get();
    qfvalue = iqFvalue.get();

    //Compute the interpolated values
    EXPECT_TRUE(dc.interpolate(fvalue, nPoints, qfvalue, nqPoints));

    {
        int order = dc.orderofAccuracy();

        std::cout << "DC-PSE uses " << dc.neighborhoodKernelSize() << " number of points in the neighborhood of each query points to do a \n"
                  << order << (order == 1 ? "st  " : order == 2 ? "nd  " : order == 3 ? "rd  " : "th  ") << "order interpolation." << std::endl;
    }

    // //Create an instance of io object
    // io file;

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_EXACT", file.in | file.out | file.trunc))
    // {
    //     data = idata.get();
    //     fvalue = iFvalue.get();
    //     for (int i = 0; i < nPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(data, 1, nDim, 2);
    //         file.saveMatrix<double>(fvalue, 1, 1);
    //         data += nDim;
    //         fvalue++;
    //     }
    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_DCPSE", file.in | file.out | file.trunc))
    // {
    //     qdata = iqdata.get();
    //     qfvalue = iqFvalue.get();
    //     for (int i = 0; i < nqPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata, 1, nDim, 2);
    //         file.saveMatrix<double>(qfvalue, 1, 1);
    //         qdata += nDim;
    //         qfvalue++;
    //     }

    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_DCPSE_EXACT", file.in | file.out | file.trunc))
    // {
    //     qdata = iqdata.get();
    //     qfvalue = iqFvalueExact.get();
    //     for (int i = 0; i < nqPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata, 1, nDim, 2);
    //         file.saveMatrix<double>(qfvalue, 1, 1);
    //         qdata += nDim;
    //         qfvalue++;
    //     }

    //     file.closeFile();
    // }
}

/*! 
 * Test to check dcpse functionality for frank2d function
 */
TEST(dcpse_test, HandlesFrank2dFunctionRandomPoints)
{
    int nDim = 2;
    double Lb[] = {0, 0};
    double Ub[] = {1, 1};
    int nDPoints[] = {21, 21};
    int nPoints = std::accumulate(nDPoints, nDPoints + nDim, 1, std::multiplies<int>());
    int nqPoints = 20;

    std::unique_ptr<double[]> idata;
    std::unique_ptr<double[]> iqdata;
    std::unique_ptr<double[]> iFvalue;
    std::unique_ptr<double[]> iqFvalue;
    std::unique_ptr<double[]> iqFvalueExact;

    double *data;
    double *qdata;
    double *fvalue;
    double *qfvalue;

    try
    {
        idata.reset(new double[nPoints * nDim]);
        iqdata.reset(new double[nqPoints * nDim]);

        iFvalue.reset(new double[nPoints]);
        iqFvalue.reset(new double[nqPoints]);
        iqFvalueExact.reset(new double[nqPoints]);
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    //Create input points
    data = idata.get();
    EXPECT_TRUE(meshgrid<double>(data, nPoints, nDim, Lb, Ub));

    //create an instance of frank2d object
    franke2d<double> f2d;

    //Compute the function value at each input point
    {
        data = idata.get();
        for (int i = 0; i < nPoints; i++)
        {
            iFvalue[i] = f2d.f(data);
            data += nDim;
        }
    }

    //Create random query data points
    {
        qdata = iqdata.get();

        // std::random_device rd;
        std::mt19937 gen(1);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < nqPoints; i++)
        {
            std::for_each(qdata, qdata + nDim, [&](double &i) { i = dis(gen); });
            qdata += nDim;
        }

        //Compute the function value at each query point
        qdata = iqdata.get();
        for (int i = 0; i < nqPoints; i++)
        {
            iqFvalueExact[i] = f2d.f(qdata);
            qdata += nDim;
        }
    }

    //Create an instance of a DC-PSE object
    dcpse<double> dc(nDim);

    data = idata.get();
    qdata = iqdata.get();

    //Compute the interpolator weights & operator kernel
    EXPECT_TRUE(dc.computeInterpolatorWeights(data, nPoints, qdata, nqPoints));

    fvalue = iFvalue.get();
    qfvalue = iqFvalue.get();

    //Compute the interpolated values
    EXPECT_TRUE(dc.interpolate(fvalue, nPoints, qfvalue, nqPoints));

    {
        int order = dc.orderofAccuracy();

        std::cout << "DC-PSE uses " << dc.neighborhoodKernelSize() << " number of points in the neighborhood of each query points to do a \n"
                  << order << (order == 1 ? "st  " : order == 2 ? "nd  " : order == 3 ? "rd  " : "th  ") << "order interpolation." << std::endl;
    }

    // //Create an instance of io object
    // io file;

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_EXACTRND", file.in | file.out | file.trunc))
    // {
    //     data = idata.get();
    //     fvalue = iFvalue.get();
    //     for (int i = 0; i < nPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(data, 1, nDim, 2);
    //         file.saveMatrix<double>(fvalue, 1, 1);
    //         data += nDim;
    //         fvalue++;
    //     }
    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_DCPSERND", file.in | file.out | file.trunc))
    // {
    //     qdata = iqdata.get();
    //     qfvalue = iqFvalue.get();
    //     for (int i = 0; i < nqPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata, 1, nDim, 2);
    //         file.saveMatrix<double>(qfvalue, 1, 1);
    //         qdata += nDim;
    //         qfvalue++;
    //     }

    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/FRANK2D_DCPSE_EXACTRND", file.in | file.out | file.trunc))
    // {
    //     qdata = iqdata.get();
    //     qfvalue = iqFvalueExact.get();
    //     for (int i = 0; i < nqPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata, 1, nDim, 2);
    //         file.saveMatrix<double>(qfvalue, 1, 1);
    //         qdata += nDim;
    //         qfvalue++;
    //     }

    //     file.closeFile();
    // }
}

/*! 
 * Test to check dcpse functionality for Rastrigin function
 */
TEST(dcpse_test, HandlesRastriginFunctionCartesianPoints)
{
    int nDim = 2;
    double Lb[] = {-5.12, -5.12};
    double Ub[] = {5.12, 5.12};
    int nDPoints[] = {21, 21};
    int nPoints = std::accumulate(nDPoints, nDPoints + nDim, 1, std::multiplies<int>());
    int nqPoints = 20;

    std::unique_ptr<double[]> idata;
    std::unique_ptr<double[]> iqdata;
    std::unique_ptr<double[]> iFvalue;
    std::unique_ptr<double[]> iqFvalue;
    std::unique_ptr<double[]> iqFvalueExact;

    double *data;
    double *qdata;
    double *fvalue;
    double *qfvalue;

    try
    {
        idata.reset(new double[nPoints * nDim]);
        iqdata.reset(new double[nqPoints * nDim]);

        iFvalue.reset(new double[nPoints]);
        iqFvalue.reset(new double[nqPoints]);
        iqFvalueExact.reset(new double[nqPoints]);
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    //Create input points
    data = idata.get();
    EXPECT_TRUE(meshgrid<double>(data, nDPoints, nDim, Lb, Ub));

    //create an instance of rastrigin 2d object
    rastrigin<double> r2d(nDim);

    //Compute the function value at each input point
    {
        data = idata.get();
        for (int i = 0; i < nPoints; i++)
        {
            iFvalue[i] = r2d.f(data);
            data += nDim;
        }
    }

    //Create random query data points
    {
        qdata = iqdata.get();

        // std::random_device rd;
        std::mt19937 gen(1);
        std::uniform_real_distribution<> dis(-5.12, 5.12);

        for (int i = 0; i < nqPoints; i++)
        {
            std::for_each(qdata, qdata + nDim, [&](double &i) { i = dis(gen); });
            qdata += nDim;
        }

        //Compute the function value at each query point
        qdata = iqdata.get();
        for (int i = 0; i < nqPoints; i++)
        {
            iqFvalueExact[i] = r2d.f(qdata);
            qdata += nDim;
        }
    }

    //Create an instance of a DC-PSE object
    dcpse<double> dc(nDim);

    data = idata.get();
    qdata = iqdata.get();

    //Compute the interpolator weights & operator kernel
    EXPECT_TRUE(dc.computeInterpolatorWeights(data, nPoints, qdata, nqPoints, 4));

    fvalue = iFvalue.get();
    qfvalue = iqFvalue.get();

    //Compute the interpolated values
    EXPECT_TRUE(dc.interpolate(fvalue, nPoints, qfvalue, nqPoints));

    {
        int order = dc.orderofAccuracy();

        std::cout << "DC-PSE uses " << dc.neighborhoodKernelSize() << " number of points in the neighborhood of each query points to do a \n"
                  << order << (order == 1 ? "st  " : order == 2 ? "nd  " : order == 3 ? "rd  " : "th  ") << "order interpolation." << std::endl;
    }

    // //Create an instance of io object
    // io file;

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/RASTRIGIN2D_EXACT", file.in | file.out | file.trunc))
    // {
    //     data = idata.get();
    //     fvalue = iFvalue.get();
    //     for (int i = 0; i < nPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(data, 1, nDim, 2);
    //         file.saveMatrix<double>(fvalue, 1, 1);
    //         data += nDim;
    //         fvalue++;
    //     }
    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/RASTRIGIN2D_DCPSE", file.in | file.out | file.trunc))
    // {
    //     qdata = iqdata.get();
    //     qfvalue = iqFvalue.get();
    //     for (int i = 0; i < nqPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata, 1, nDim, 2);
    //         file.saveMatrix<double>(qfvalue, 1, 1);
    //         qdata += nDim;
    //         qfvalue++;
    //     }

    //     file.closeFile();
    // }

    // //!Open a file for reading and writing
    // if (file.openFile("./dcpse/RASTRIGIN2D_DCPSE_EXACT", file.in | file.out | file.trunc))
    // {
    //     qdata = iqdata.get();
    //     qfvalue = iqFvalueExact.get();
    //     for (int i = 0; i < nqPoints; i++)
    //     {
    //         //!Write the matrix in it
    //         file.saveMatrix<double>(qdata, 1, nDim, 2);
    //         file.saveMatrix<double>(qfvalue, 1, 1);
    //         qdata += nDim;
    //         qfvalue++;
    //     }

    //     file.closeFile();
    // }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
