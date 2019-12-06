
#include "numerics/linearregression.hpp"
#include "numerics/fitness.hpp"
#include "numerics/testfunctions/predictiontestfunctions.hpp"
#include "io/io.hpp"
#include "gtest/gtest.h"

#include <random>

#define WRITE_TO_FILE 0

//Data container
std::unique_ptr<double[]> idata;
std::unique_ptr<double[]> iqdata;
std::unique_ptr<double[]> iFvalue;
std::unique_ptr<double[]> iqFvalue;
std::unique_ptr<double[]> iqFvalueExact;

//Dummy pointer
double *data;
double *qdata;
double *fvalue;
double *qfvalue;

template <typename T>
void fillPagebyPage(T *inDataPt, T *coords, int const d, T lx, T ly, T dx, T dy, int x, int y)
{
    data = inDataPt;
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

/*!
 * \ingroup Test_Module
 *
 * \brief Compute N-D grid coordinates between \f$ [Lb \cdots Ub] \f$
 *
 * \tparam T        data type
 *
 * \param inDataPt  N-D grid coordinates
 * \param nDPoints  Number of points in each direction
 * \param nDim      Dimensionality
 * \param Lb        Lower bound in each dimension
 * \param Ub        Upper bound in each dimension
 *
 * \return true     if it successfully creates N-D grid coordinates
 */
template <typename T>
bool meshgrid(T *&inDataPt, int const *nDPoints, int const nDim, double *Lb, double *Ub)
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

    if (inDataPt == nullptr)
    {
        try
        {
            inDataPt = new T[nPoints * nDim];
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
            inDataPt[i] = Lb[0] + i * dx[0];
        }
        break;
    case 2:
    {
        for (int i = 0, n = 0; i < nDPoints[0]; i++)
        {
            T const r = Lb[0] + i * dx[0];
            for (int j = 0; j < nDPoints[1]; j++)
            {
                inDataPt[n] = r;
                inDataPt[n + 1] = Lb[1] + j * dx[1];
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

        //     d = inDataPt + counter;

        //     fillPagebyPage<T>(d, coords, nd2, Lb[nd2], Lb[nd1], dx[nd2], dx[nd1], nDPoints[nd2], nDPoints[nd1]);

        //     counter += nDim * nDPoints[nd2] * nDPoints[nd1];
        // }

        // delete[] coords;
        break;
    }

    delete[] dx;
    return true;
}

/*!
 * \ingroup Test_Module
 *
 * \brief Compute N-D coordinates randomly distributed between \f$ [Lb \cdots Ub] \f$
 *
 * \tparam T        data type
 *
 * \param inDataPt     N-D coordinates
 * \param nPoints   Total number of points
 * \param nDim      Dimensionality
 * \param Lb        Lower bound in each dimension
 * \param Ub        Upper bound in each dimension
 *
 * \return true     if it successfully creates N-D grid coordinates
 */
template <typename T>
bool meshgrid(T *&inDataPt, int const nPoints, int const nDim, double *Lb, double *Ub)
{
    if (nDim < 1)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Wrong dimension ! " << std::endl;
        return false;
    }

    if (inDataPt == nullptr)
    {
        try
        {
            inDataPt = new T[nPoints * nDim];
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
            inDataPt[i] = Lb[0] + dis(gen) * (Ub[0] - Lb[0]);
        }
        break;

    case 2:
        for (int i = 0, n = 0; i < nPoints; i++)
        {
            inDataPt[n] = Lb[0] + dis(gen) * (Ub[0] - Lb[0]);
            inDataPt[n + 1] = Lb[1] + dis(gen) * (Ub[1] - Lb[1]);
            n += 2;
        }
        break;

    default:
        for (int i = 0, n = 0; i < nPoints; i++)
        {
            for (int d = 0; d < nDim; d++, n++)
            {
                inDataPt[n] = Lb[d] + dis(gen) * (Ub[d] - Lb[d]);
            }
        }
        break;
    }

    return true;
}

/*!
 * \ingroup Test_Module
 *
 * Test to check linearregression functionality for Qian function
 */
TEST(linearregression_1d, HandlesQianFunction)
{
    //! Dimension
    int nDim = 1;
    //! Bounds
    double Lb[] = {0};
    double Ub[] = {1};
    //! Number of points in each direction
    int nDPoints[] = {10};
    //! Total number of training points
    int nPoints = std::accumulate(nDPoints, nDPoints + nDim, 1, std::multiplies<int>());
    //! Number of query points
    int nqPoints = 30;

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

    //Create an instance of Qian object
    qian<double> Qian;

    {
        //Create input points
        data = idata.get();
        EXPECT_TRUE(meshgrid<double>(data, nDPoints, nDim, Lb, Ub));

        //Compute the function value at each input point
        data = idata.get();
        for (int i = 0; i < nPoints; i++)
        {
            iFvalue[i] = Qian.f(data);
            data += nDim;
        }
    }

    {
        // std::random_device rd;
        std::mt19937 gen(1);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        //Create random query data points
        qdata = iqdata.get();
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

    {
        //Create an instance of a Linear Regression object
        umuq::linearRegression<double> lr(nDim, 1);

        data = idata.get();
        fvalue = iFvalue.get();

        //Compute the interpolator weights & operator kernel
        EXPECT_TRUE(lr.computeWeights(data, fvalue, nPoints));

        qdata = iqdata.get();
        qfvalue = iqFvalue.get();

        //Compute the interpolated values
        EXPECT_TRUE(lr.solve(qdata, qfvalue, nqPoints));

        {
            umuq::fitness<double> f("root_mean_squared");
            fvalue = iqFvalueExact.get();
            qfvalue = iqFvalue.get();
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(fvalue, qfvalue, nqPoints) << std::endl;
        }

        //Writing to external files
        if (WRITE_TO_FILE)
        {
            //Create an instance of io object
            umuq::io file;

            //!Open a file for reading and writing
            if (file.openFile("./regression/QIAN_TRAIN", file.in | file.out | file.trunc))
            {
                data = idata.get();
                fvalue = iFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(data, nDim, fvalue, 1, nPoints);
                file.closeFile();
            }

            //!Open a file for reading and writing
            if (file.openFile("./regression/QIAN_LR_1", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }

            //!Open a file for reading and writing
            if (file.openFile("./regression/QIAN_EXACT", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalueExact.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }
        }
    }

    {
        //Create an instance of a Linear Regression object
        umuq::linearRegression<double> lr(nDim, 2);

        data = idata.get();
        fvalue = iFvalue.get();

        //Compute the interpolator weights & operator kernel
        EXPECT_TRUE(lr.computeWeights(data, fvalue, nPoints));

        qdata = iqdata.get();
        qfvalue = iqFvalue.get();

        //Compute the interpolated values
        EXPECT_TRUE(lr.solve(qdata, qfvalue, nqPoints));

        {
            umuq::fitness<double> f("root_mean_squared");
            fvalue = iqFvalueExact.get();
            qfvalue = iqFvalue.get();
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(fvalue, qfvalue, nqPoints) << std::endl;
        }

        //Writing to external files
        if (WRITE_TO_FILE)
        {
            //Create an instance of io object
            umuq::io file;

            //!Open a file for reading and writing
            if (file.openFile("./regression/QIAN_LR_2", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }
        }
    }

    {
        //Create an instance of a Linear Regression object
        umuq::linearRegression<double> lr(nDim, 3);

        data = idata.get();
        fvalue = iFvalue.get();

        //Compute the interpolator weights & operator kernel
        EXPECT_TRUE(lr.computeWeights(data, fvalue, nPoints));

        qdata = iqdata.get();
        qfvalue = iqFvalue.get();

        //Compute the interpolated values
        EXPECT_TRUE(lr.solve(qdata, qfvalue, nqPoints));

        {
            umuq::fitness<double> f("root_mean_squared");
            fvalue = iqFvalueExact.get();
            qfvalue = iqFvalue.get();
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(fvalue, qfvalue, nqPoints) << std::endl;
        }

        //Writing to external files
        if (WRITE_TO_FILE)
        {
            //Create an instance of io object
            umuq::io file;

            //!Open a file for reading and writing
            if (file.openFile("./regression/QIAN_LR_3", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }
        }
    }

    {
        //Create an instance of a Linear Regression object
        umuq::linearRegression<double> lr(nDim, 4);

        data = idata.get();
        fvalue = iFvalue.get();

        //Compute the interpolator weights & operator kernel
        EXPECT_TRUE(lr.computeWeights(data, fvalue, nPoints));

        qdata = iqdata.get();
        qfvalue = iqFvalue.get();

        //Compute the interpolated values
        EXPECT_TRUE(lr.solve(qdata, qfvalue, nqPoints));

        {
            umuq::fitness<double> f("root_mean_squared");
            fvalue = iqFvalueExact.get();
            qfvalue = iqFvalue.get();
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(fvalue, qfvalue, nqPoints) << std::endl;
        }

        //Writing to external files
        if (WRITE_TO_FILE)
        {
            //Create an instance of io object
            umuq::io file;

            //!Open a file for reading and writing
            if (file.openFile("./regression/QIAN_LR_4", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }
        }
    }
}

/*!
 * \ingroup Test_Module
 *
 * Test to check linearregression functionality for CFD results
 */
TEST(linearregression_1d, HandlesCFDResults)
{
    //! Dimension
    int nDim = 1;
    //! Bounds
    // double Lb[] = {-4};
    // double Ub[] = {1.2};
    //! Number of points in each direction
    int nDPoints[] = {9};
    //! Total number of training points
    int nPoints = std::accumulate(nDPoints, nDPoints + nDim, 1, std::multiplies<int>());
    //! Number of query points
    int nqPoints = 27;

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

    {
        //Create input points
        data = idata.get();
        for (int i = 0; i < nPoints; ++i)
        {
            data[i] = 1. - i * 0.6;
        }

        double FVALUE[] = {0.001399, 0.002119, 0.000223, 0.000265, 0.001145, -0.001141, -0.002576, -0.004553, -0.005357};

        fvalue = iFvalue.get();
        std::copy(FVALUE, FVALUE + nPoints, fvalue);
    }

    {
        //Create random query data points
        qdata = iqdata.get();
        for (int i = 0; i < nqPoints; i++)
        {
            qdata[i] = -4. + i * 0.2;
        }

        //Get the function value at each query point
        double FVALUE[] = {-4.86257420168973407e-03, -5.35699999999999978e-03, -5.48294572673046956e-03, -5.18719553406514142e-03, -4.55299999999999976e-03, -3.77857337305603086e-03,
                           -3.08158623047766591e-03, -2.57599999999999973e-03, -2.20345834008881187e-03, -1.77996424185716470e-03, -1.14099999999999879e-03, -2.88267401909034519e-04,
                           5.73070180457921569e-04, 1.14500000000000040e-03, 1.22525475319120000e-03, 8.45239315114444537e-04, 2.65000000000000912e-04, -1.67025267563533728e-04,
                           -1.95990586912018171e-04, 2.23000000000003577e-04, 9.26826735582121424e-04, 1.64304202219159681e-03, 2.11900000000000456e-03, 2.21642368852469247e-03,
                           1.93903577793026120e-03, 1.39900000000000462e-03, 7.52330802551779646e-04};

        fvalue = iqFvalueExact.get();
        std::copy(FVALUE, FVALUE + nqPoints, fvalue);
    }

    {
        //Create an instance of a Linear Regression object
        umuq::linearRegression<double> lr(nDim, 1);

        data = idata.get();
        fvalue = iFvalue.get();

        //Compute the interpolator weights & operator kernel
        EXPECT_TRUE(lr.computeWeights(data, fvalue, nPoints));

        qdata = iqdata.get();
        qfvalue = iqFvalue.get();

        //Compute the interpolated values
        EXPECT_TRUE(lr.solve(qdata, qfvalue, nqPoints));

        {
            umuq::fitness<double> f("root_mean_squared");
            fvalue = iqFvalueExact.get();
            qfvalue = iqFvalue.get();
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(fvalue, qfvalue, nqPoints) << std::endl;
        }

        //Writing to external files
        if (WRITE_TO_FILE)
        {
            //Create an instance of io object
            umuq::io file;

            //!Open a file for reading and writing
            if (file.openFile("./regression/CFD_TRAIN", file.in | file.out | file.trunc))
            {
                data = idata.get();
                fvalue = iFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(data, nDim, fvalue, 1, nPoints);
                file.closeFile();
            }

            //!Open a file for reading and writing
            if (file.openFile("./regression/CFD_LR_1", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }

            //!Open a file for reading and writing
            if (file.openFile("./regression/CFD_EXACT", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalueExact.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }
        }
    }

    {
        //Create an instance of a Linear Regression object
        umuq::linearRegression<double> lr(nDim, 2);

        data = idata.get();
        fvalue = iFvalue.get();

        //Compute the interpolator weights & operator kernel
        EXPECT_TRUE(lr.computeWeights(data, fvalue, nPoints));

        qdata = iqdata.get();
        qfvalue = iqFvalue.get();

        //Compute the interpolated values
        EXPECT_TRUE(lr.solve(qdata, qfvalue, nqPoints));

        {
            umuq::fitness<double> f("root_mean_squared");
            fvalue = iqFvalueExact.get();
            qfvalue = iqFvalue.get();
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(fvalue, qfvalue, nqPoints) << std::endl;
        }

        //Writing to external files
        if (WRITE_TO_FILE)
        {
            //Create an instance of io object
            umuq::io file;

            //!Open a file for reading and writing
            if (file.openFile("./regression/CFD_LR_2", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }
        }
    }

    {
        //Create an instance of a Linear Regression object
        umuq::linearRegression<double> lr(nDim, 3);

        data = idata.get();
        fvalue = iFvalue.get();

        //Compute the interpolator weights & operator kernel
        EXPECT_TRUE(lr.computeWeights(data, fvalue, nPoints));

        qdata = iqdata.get();
        qfvalue = iqFvalue.get();

        //Compute the interpolated values
        EXPECT_TRUE(lr.solve(qdata, qfvalue, nqPoints));

        {
            umuq::fitness<double> f("root_mean_squared");
            fvalue = iqFvalueExact.get();
            qfvalue = iqFvalue.get();
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(fvalue, qfvalue, nqPoints) << std::endl;
        }

        //Writing to external files
        if (WRITE_TO_FILE)
        {
            //Create an instance of io object
            umuq::io file;

            //!Open a file for reading and writing
            if (file.openFile("./regression/CFD_LR_3", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }
        }
    }

    {
        //Create an instance of a Linear Regression object
        umuq::linearRegression<double> lr(nDim, 4);

        data = idata.get();
        fvalue = iFvalue.get();

        //Compute the interpolator weights & operator kernel
        EXPECT_TRUE(lr.computeWeights(data, fvalue, nPoints));

        qdata = iqdata.get();
        qfvalue = iqFvalue.get();

        //Compute the interpolated values
        EXPECT_TRUE(lr.solve(qdata, qfvalue, nqPoints));

        {
            umuq::fitness<double> f("root_mean_squared");
            fvalue = iqFvalueExact.get();
            qfvalue = iqFvalue.get();
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(fvalue, qfvalue, nqPoints) << std::endl;
        }

        //Writing to external files
        if (WRITE_TO_FILE)
        {
            //Create an instance of io object
            umuq::io file;

            //!Open a file for reading and writing
            if (file.openFile("./regression/CFD_LR_4", file.in | file.out | file.trunc))
            {
                qdata = iqdata.get();
                qfvalue = iqFvalue.get();
                //!Write the matrix in it
                file.saveMatrix<double>(qdata, nDim, qfvalue, 1, nqPoints);
                file.closeFile();
            }
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
