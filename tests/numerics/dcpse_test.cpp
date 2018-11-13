#include "core/core.hpp"
#include "environment.hpp"
#include "io/io.hpp"
#include "numerics/hypercube/hypercubesampling.hpp"
#include "numerics/dcpse.hpp"
#include "numerics/fitness.hpp"
#include "numerics/testfunctions/predictiontestfunctions.hpp"
#include "numerics/stats.hpp"
#include "gtest/gtest.h"

#define WRITE_TO_FILE 0

/*!
 * \ingroup Test_Module
 * 
 * \brief Get an instance of a seeded double random object
 * 
 */
umuq::psrandom<double> prng(123);

/*!
 * \ingroup Test_Module
 *
 * Test to check dcpse functionality for Qian function
 */
TEST(dcpse_1d, HandlesQianFunction)
{
    // Problem dimension
    int numDimensions = 1;

    // Number of points in each direction
    std::vector<int> numPointsInEachDirection{10};

    // Number of data points
    int const numDataPoints = std::accumulate(numPointsInEachDirection.begin(), numPointsInEachDirection.end(), 1, std::multiplies<int>());

    // Number of query points
    int const numQueryPoints = 30;

    std::vector<double> dataPoints(numDataPoints * numDimensions);
    std::vector<double> dataFuncationValues(numDataPoints);
    std::vector<double> queryPoints(numQueryPoints * numDimensions);
    std::vector<double> queryFuncationValues(numQueryPoints);
    std::vector<double> exactQueryFuncationValues(numQueryPoints);

    // Create an instance of Qian object
    qian<double> Qian;

    {
        // Create an instance of the hypercubeSampling object in unit hypercube
        umuq::hypercubeSampling<double> Domain(numPointsInEachDirection);

        // Create input points
        EXPECT_TRUE(Domain.grid(dataPoints));

        // Compute the function value at each input point
        double *data = dataPoints.data();
        for (int i = 0; i < numDataPoints; i++)
        {
            dataFuncationValues[i] = Qian.f(data);
            data += numDimensions;
        }

        // Initialize the PRNG or set the state of the PRNG
        EXPECT_TRUE(prng.setState());

        // Set the pseudo random number generator
        EXPECT_TRUE(Domain.setRandomGenerator(&prng));

        // Create uniformly random distributed input points
        EXPECT_TRUE(Domain.sample(queryPoints, numQueryPoints));

        // Compute the function value at each query point
        data = queryPoints.data();
        for (int i = 0; i < numQueryPoints; i++)
        {
            exactQueryFuncationValues[i] = Qian.f(data);
            data += numDimensions;
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 2));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/QIAN_TRAIN", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(dataPoints, numDimensions, dataFuncationValues, 1, numDataPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/QIAN_DCPSE_2", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/QIAN_EXACT", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, exactQueryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 3));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/QIAN_DCPSE_3", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }
}

/*!
 * \ingroup Test_Module
 *
 * Test to check dcpse functionality for CFD results
 */
TEST(dcpse_1d, HandlesCFDResults)
{
    // Problem dimension
    int numDimensions = 1;

    // Number of points in each direction
    std::vector<int> numPointsInEachDirection{9};

    // Number of data points
    int const numDataPoints = 9;

    //  Number of query points
    int const numQueryPoints = 27;

    std::vector<double> dataPoints(numDataPoints * numDimensions);
    std::vector<double> dataFuncationValues{0.001399, 0.002119, 0.000223, 0.000265, 0.001145, -0.001141, -0.002576, -0.004553, -0.005357};
    std::vector<double> queryPoints(numQueryPoints * numDimensions);
    std::vector<double> queryFuncationValues{-4.86257420168973407e-03, -5.35699999999999978e-03, -5.48294572673046956e-03,
                                             -5.18719553406514142e-03, -4.55299999999999976e-03, -3.77857337305603086e-03,
                                             -3.08158623047766591e-03, -2.57599999999999973e-03, -2.20345834008881187e-03,
                                             -1.77996424185716470e-03, -1.14099999999999879e-03, -2.88267401909034519e-04,
                                             5.73070180457921569e-04, 1.14500000000000040e-03, 1.22525475319120000e-03,
                                             8.45239315114444537e-04, 2.65000000000000912e-04, -1.67025267563533728e-04,
                                             -1.95990586912018171e-04, 2.23000000000003577e-04, 9.26826735582121424e-04,
                                             1.64304202219159681e-03, 2.11900000000000456e-03, 2.21642368852469247e-03,
                                             1.93903577793026120e-03, 1.39900000000000462e-03, 7.52330802551779646e-04};
    std::vector<double> exactQueryFuncationValues(numQueryPoints);

    // Data points
    for (int i = 0; i < numDataPoints; ++i)
    {
        dataPoints[i] = 1. - i * 0.6;
    }

    // Query data points
    for (int i = 0; i < numQueryPoints; i++)
    {
        queryPoints[i] = -4. + i * 0.2;
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 2));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/CFD_TRAIN", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(dataPoints, numDimensions, dataFuncationValues, 1, numDataPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/CFD_DCPSE_2", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/CFD_KRIGING", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 3));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/CFD_DCPSE_3", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }
}

/*!
 * \ingroup Test_Module
 *
 * Test to check dcpse functionality for Matlab Peaks function
 */
TEST(dcpse_2d, HandlesPeaksFunction)
{
    // Problem dimension
    int numDimensions = 2;

    // Number of points in each direction
    std::vector<int> numPointsInEachDirection{25, 25};

    // Number of data points
    int const numDataPoints = std::accumulate(numPointsInEachDirection.begin(), numPointsInEachDirection.end(), 1, std::multiplies<int>());

    // Domain bounds
    std::vector<double> LowerBounds{-3, -3};
    std::vector<double> UpperBounds{3, 3};

    //  Number of query points
    int const numQueryPoints = 150;

    std::vector<double> dataPoints(numDataPoints * numDimensions);
    std::vector<double> dataFuncationValues(numDataPoints);
    std::vector<double> queryPoints(numQueryPoints * numDimensions);
    std::vector<double> queryFuncationValues(numQueryPoints);
    std::vector<double> exactQueryFuncationValues(numQueryPoints);

    // Create an instance of peaks object
    peaks<double> Peaks;

    {
        // Create an instance of the hypercubeSampling object in unit hypercube
        umuq::hypercubeSampling<double> Domain(numPointsInEachDirection, LowerBounds, UpperBounds);

        // Create input points
        EXPECT_TRUE(Domain.grid(dataPoints));

        // Compute the function value at each input point
        double *data = dataPoints.data();
        for (int i = 0; i < numDataPoints; i++)
        {
            dataFuncationValues[i] = Peaks.f(data);
            data += numDimensions;
        }

        // Initialize the PRNG or set the state of the PRNG
        EXPECT_TRUE(prng.setState());

        // Set the pseudo random number generator
        EXPECT_TRUE(Domain.setRandomGenerator(&prng));

        // Create uniformly random distributed points in the hypercube
        EXPECT_TRUE(Domain.sample(queryPoints, numQueryPoints));

        // Compute the function value at each query point
        data = queryPoints.data();
        for (int i = 0; i < numQueryPoints; i++)
        {
            exactQueryFuncationValues[i] = Peaks.f(data);
            data += numDimensions;
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 2));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/PEAKS_TRAIN", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(dataPoints, numDimensions, dataFuncationValues, 1, numDataPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/PEAKS_DCPSE_2", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/PEAKS_EXACT", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, exactQueryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 3));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/PEAKS_DCPSE_3", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }
}

/*!
 * \ingroup Test_Module
 *
 * Test to check dcpse functionality for Matlab Peaks function
 */
TEST(dcpse_2d, HandlesPeaksRndFunction)
{
    // Problem dimension
    int numDimensions = 2;

    // Number of points in each direction
    std::vector<int> numPointsInEachDirection{25, 25};

    // Number of data points
    int const numDataPoints = std::accumulate(numPointsInEachDirection.begin(), numPointsInEachDirection.end(), 1, std::multiplies<int>());

    // Domain bounds
    std::vector<double> LowerBounds{-3, -3};
    std::vector<double> UpperBounds{3, 3};

    //  Number of query points
    int const numQueryPoints = 169;

    std::vector<double> dataPoints(numDataPoints * numDimensions);
    std::vector<double> dataFuncationValues(numDataPoints);
    // dataPoints{1.09624842999999994e+00, -2.75270990000000015e+00, 8.30415631000000043e-01, 2.50714771999999986e+00, -5.58123205000000011e-01, -1.04478639000000006e+00,
    //            -7.38208187999999987e-01, -1.52036992999999998e+00, 2.63411796000000020e+00, 1.64582070999999996e+00, -2.35162979999999999e+00, -1.89140579000000009e+00,
    //            1.51688767000000002e+00, -8.57453590999999959e-01, -2.70541212000000009e+00, 1.37300754999999993e-02, -2.44701633000000002e-01, 1.42774384999999993e+00,
    //            5.54149462000000023e-02, 6.44981041999999949e-01, -1.20377540999999999e+00, 2.31058493000000009e+00, -1.57823373999999994e+00, -1.59712943999999996e-01,
    //            -1.93072248000000002e+00, 5.61778499999999958e-01, 2.36509929999999979e+00, 2.70111692000000003e+00, 4.52539931999999978e-01, -1.25023861000000003e+00,
    //            -1.10169632000000006e+00, 1.89351391999999996e+00, 2.84759918000000001e+00, -2.62754845999999986e+00, 1.31288841000000001e+00, -4.53010866999999984e-01,
    //            -2.65530153999999996e+00, -2.23173899000000020e+00, 1.91541044999999999e+00, 9.77562617000000023e-01};
    // dataFuncationValues{-2.41176651999999991e-01, 9.29606946000000045e-01, 2.01766673000000019e+00, -8.36704785999999978e-01, 1.92744452000000004e-02,
    //                     1.81541277999999993e-02, 1.38611237999999992e+00, -1.36023139999999987e-01, 7.30667539999999960e+00, 7.68382919999999969e-01,
    //                     7.24030752999999971e-01, -2.33695396000000022e+00, -1.23299957999999998e+00, 3.95091529999999982e-03, -4.53071206999999987e+00,
    //                     1.90473673999999993e+00, -9.22950295000000005e-05, 2.91661564999999978e+00, 3.09315891000000020e-03, 7.40574621000000044e-01};

    std::vector<double> queryPoints(numQueryPoints * numDimensions);
    std::vector<double> queryFuncationValues(numQueryPoints);
    std::vector<double> exactQueryFuncationValues(numQueryPoints);
    // exactQueryFuncationValues{-9.701095e-01, -8.072969e-01, -6.925640e-01, -6.161840e-01, -5.657533e-01, -5.272067e-01, -4.860000e-01, -4.283220e-01, -3.421956e-01, -2.183451e-01, -5.074541e-02, 1.631880e-01, 4.227573e-01, -1.059100e-01, 3.823685e-02, 1.130861e-01, 1.290740e-01,
    //                           1.005853e-01, 4.484181e-02, -1.948642e-02, -7.379033e-02, -1.009816e-01, -8.674388e-02, -2.044937e-02, 1.043343e-01, 2.897685e-01, -1.248149e+00, -1.061934e+00, -9.358554e-01, -8.669999e-01, -8.486016e-01, -8.702113e-01, -9.182153e-01,
    //                           -9.766757e-01, -1.028423e+00, -1.056304e+00, -1.044456e+00, -9.794795e-01, -8.513920e-01, -3.930977e+00, -3.629378e+00, -3.333798e+00, -3.053884e+00, -2.798622e+00, -2.574931e+00, -2.386389e+00, -2.232290e+00, -2.107210e+00, -2.001193e+00,
    //                           -1.900551e+00, -1.789205e+00, -1.650379e+00, -5.701791e+00, -4.865644e+00, -3.906274e+00, -2.854210e+00, -1.748517e+00, -6.335483e-01, 4.450367e-01, 1.444495e+00, 2.328935e+00, 3.072478e+00, 3.661247e+00, 4.093985e+00, 4.381275e+00,
    //                           -3.145570e+00, -1.933898e+00, -5.749548e-01, 8.938113e-01, 2.423906e+00, 3.959092e+00, 5.439698e+00, 6.807465e+00, 8.010410e+00, 9.007121e+00, 9.770002e+00, 1.028711e+01, 1.056245e+01, -3.889753e+00, -3.302587e+00, -2.628603e+00,
    //                           -1.876864e+00, -1.062555e+00, -2.061836e-01, 6.678065e-01, 1.532884e+00, 2.362557e+00, 3.132528e+00, 3.822653e+00, 4.418461e+00, 4.912098e+00, -4.941672e+00, -4.806323e+00, -4.641897e+00, -4.443134e+00, -4.205943e+00, -3.927587e+00,
    //                           -3.606789e+00, -3.243751e+00, -2.840104e+00, -2.398780e+00, -1.923837e+00, -1.420238e+00, -8.936063e-01, -9.722428e-01, -3.897467e-01, 1.747136e-01, 7.053661e-01, 1.187746e+00, 1.610173e+00, 1.965003e+00, 2.249483e+00, 2.466094e+00,
    //                           2.622317e+00, 2.729862e+00, 2.803465e+00, 2.859406e+00, -1.607513e+00, -8.977283e-01, -1.623442e-01, 5.781354e-01, 1.301221e+00, 1.984380e+00, 2.607094e+00, 3.152763e+00, 3.610234e+00, 3.974746e+00, 4.248180e+00, 4.438619e+00,
    //                           4.559272e+00, -3.064726e+00, -2.670523e+00, -2.250940e+00, -1.813505e+00, -1.367569e+00, -9.233397e-01, -4.907562e-01, -7.835923e-02, 3.076759e-01, 6.642098e-01, 9.914151e-01, 1.292624e+00, 1.573783e+00, -1.197362e+00, -9.792264e-01,
    //                           -7.994901e-01, -6.574329e-01, -5.500605e-01, -4.719834e-01, -4.154862e-01, -3.708161e-01, -3.266908e-01, -2.710012e-01, -1.916542e-01, -7.747426e-02, 8.092907e-02, -7.961178e-02, 1.846539e-01, 4.027286e-01, 5.765432e-01, 7.106224e-01,
    //                           8.119013e-01, 8.893352e-01, 9.533332e-01, 1.015063e+00, 1.085685e+00, 1.175594e+00, 1.293728e+00, 1.447019e+00};

    // Create an instance of peaks object
    peaks<double> Peaks;

    {
        // Create an instance of the hypercubeSampling object
        umuq::hypercubeSampling<double> Domain(numDataPoints, numDimensions, LowerBounds.data(), UpperBounds.data());

        // Initialize the PRNG or set the state of the PRNG
        EXPECT_TRUE(prng.setState());

        // Set the pseudo random number generator
        EXPECT_TRUE(Domain.setRandomGenerator(&prng));

        // Create uniformly random distributed points in the hypercube
        EXPECT_TRUE(Domain.sample(dataPoints));

        // Compute the function value at each input point
        double *data = dataPoints.data();
        for (int i = 0; i < numDataPoints; i++)
        {
            dataFuncationValues[i] = Peaks.f(data);
            data += numDimensions;
        }

        // Create uniformly random distributed points in the hypercube
        EXPECT_TRUE(Domain.sample(queryPoints, numQueryPoints));

        // Compute the function value at each query point
        data = queryPoints.data();
        for (int i = 0; i < numQueryPoints; i++)
        {
            exactQueryFuncationValues[i] = Peaks.f(data);
            data += numDimensions;
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 2));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/PEAKS_RND_TRAIN", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(dataPoints, numDimensions, dataFuncationValues, 1, numDataPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/PEAKS_RND_DCPSE_2", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/PEAKS_RND_EXACT", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, exactQueryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 3));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/PEAKS_RND_DCPSE_3", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }
}

/*!
 * \ingroup Test_Module
 *
 * Test to check dcpse functionality for franke2d function
 */
TEST(dcpse_2d, HandlesFrankFunction)
{
    // Problem dimension
    int numDimensions = 2;

    // Number of points in each direction
    std::vector<int> numPointsInEachDirection{25, 25};

    // Number of data points
    int const numDataPoints = std::accumulate(numPointsInEachDirection.begin(), numPointsInEachDirection.end(), 1, std::multiplies<int>());

    //  Number of query points
    int const numQueryPoints = 150;

    std::vector<double> dataPoints(numDataPoints * numDimensions);
    std::vector<double> dataFuncationValues(numDataPoints);
    std::vector<double> queryPoints(numQueryPoints * numDimensions);
    std::vector<double> queryFuncationValues(numQueryPoints);
    std::vector<double> exactQueryFuncationValues(numQueryPoints);

    // Create an instance of Frank2d object
    franke2d<double> Frank2d;

    {
        // Create an instance of the hypercubeSampling object
        umuq::hypercubeSampling<double> Domain(numPointsInEachDirection);

        // Create input points
        EXPECT_TRUE(Domain.grid(dataPoints));

        // Compute the function value at each input point
        double *data = dataPoints.data();
        for (int i = 0; i < numDataPoints; i++)
        {
            dataFuncationValues[i] = Frank2d.f(data);
            data += numDimensions;
        }

        // Initialize the PRNG or set the state of the PRNG
        EXPECT_TRUE(prng.setState());

        // Set the pseudo random number generator
        EXPECT_TRUE(Domain.setRandomGenerator(&prng));

        // Create uniformly random distributed points in the hypercube
        EXPECT_TRUE(Domain.sample(queryPoints, numQueryPoints));

        // Compute the function value at each query point
        data = queryPoints.data();
        for (int i = 0; i < numQueryPoints; i++)
        {
            exactQueryFuncationValues[i] = Frank2d.f(data);
            data += numDimensions;
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 2));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/FRANKS2D_TRAIN", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(dataPoints, numDimensions, dataFuncationValues, 1, numDataPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/FRANKS2D_DCPSE_2", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/FRANKS2D_EXACT", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, exactQueryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 3));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/FRANKS2D_DCPSE_3", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }
}

/*!
 * \ingroup Test_Module
 *
 * Test to check dcpse functionality for franke2d function
 */
TEST(dcpse_2d, HandlesFrankRndFunction)
{
    // Problem dimension
    int numDimensions = 2;

    // Number of points in each direction
    std::vector<int> numPointsInEachDirection{25, 25};

    // Number of data points
    int const numDataPoints = std::accumulate(numPointsInEachDirection.begin(), numPointsInEachDirection.end(), 1, std::multiplies<int>());

    //  Number of query points
    int numQueryPoints = 150;

    std::vector<double> dataPoints(numDataPoints * numDimensions);
    std::vector<double> dataFuncationValues(numDataPoints);
    std::vector<double> queryPoints(numQueryPoints * numDimensions);
    std::vector<double> queryFuncationValues(numQueryPoints);
    std::vector<double> exactQueryFuncationValues(numQueryPoints);

    // Create an instance of Frank2d object
    franke2d<double> Frank2d;

    {
        // Create an instance of the hypercubeSampling object in unit hypercube
        umuq::hypercubeSampling<double> Domain(numDataPoints, numDimensions);

        // Initialize the PRNG or set the state of the PRNG
        EXPECT_TRUE(prng.setState());

        // Set the pseudo random number generator
        EXPECT_TRUE(Domain.setRandomGenerator(&prng));

        // Create uniformly random distributed points in the hypercube
        EXPECT_TRUE(Domain.sample(dataPoints));

        // Compute the function value at each input point
        double *data = dataPoints.data();
        for (int i = 0; i < numDataPoints; i++)
        {
            dataFuncationValues[i] = Frank2d.f(data);
            data += numDimensions;
        }

        // Create uniformly random distributed points in the hypercube
        EXPECT_TRUE(Domain.sample(queryPoints, numQueryPoints));

        // Compute the function value at each query point
        data = queryPoints.data();
        for (int i = 0; i < numQueryPoints; i++)
        {
            exactQueryFuncationValues[i] = Frank2d.f(data);
            data += numDimensions;
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 2));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/FRANKS2D_RND_TRAIN", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(dataPoints, numDimensions, dataFuncationValues, 1, numDataPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/FRANKS2D_RND_DCPSE_2", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/FRANKS2D_RND_EXACT", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, exactQueryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 3));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/FRANKS2D_RND_DCPSE_3", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }
}

/*!
 * \ingroup Test_Module
 *
 * Test to check dcpse functionality for Rastrigin function
 */
TEST(dcpse_2d, HandlesRastriginFunction)
{
    // Problem dimension
    int numDimensions = 2;

    // Number of points in each direction
    std::vector<int> numPointsInEachDirection{100, 100};

    // Number of data points
    int const numDataPoints = std::accumulate(numPointsInEachDirection.begin(), numPointsInEachDirection.end(), 1, std::multiplies<int>());

    // Domain bounds
    std::vector<double> LowerBounds{-5.12, -5.12};
    std::vector<double> UpperBounds{5.12, 5.12};

    //  Number of query points
    int const numQueryPoints = 150;

    std::vector<double> dataPoints(numDataPoints * numDimensions);
    std::vector<double> dataFuncationValues(numDataPoints);
    std::vector<double> queryPoints(numQueryPoints * numDimensions);
    std::vector<double> queryFuncationValues(numQueryPoints);
    std::vector<double> exactQueryFuncationValues(numQueryPoints);

    // Create an instance of Rastrigin object
    rastrigin<double> Rastrigin;

    {
        // Create an instance of the hypercubeSampling object in an unite hypercube
        umuq::hypercubeSampling<double> Domain(numPointsInEachDirection, LowerBounds, UpperBounds);

        // Create input points
        EXPECT_TRUE(Domain.grid(dataPoints));

        // Compute the function value at each input point
        double *data = dataPoints.data();
        for (int i = 0; i < numDataPoints; i++)
        {
            dataFuncationValues[i] = Rastrigin.f(data);
            data += numDimensions;
        }

        // Initialize the PRNG or set the state of the PRNG
        EXPECT_TRUE(prng.setState());

        // Set the pseudo random number generator
        EXPECT_TRUE(Domain.setRandomGenerator(&prng));

        // Create uniformly random distributed points in the hypercube
        EXPECT_TRUE(Domain.sample(queryPoints, numQueryPoints));

        // Compute the function value at each query point
        data = queryPoints.data();
        for (int i = 0; i < numQueryPoints; i++)
        {
            exactQueryFuncationValues[i] = Rastrigin.f(data);
            data += numDimensions;
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 2));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        {
            umuq::fitness<double> f("root_mean_squared");
            std::cout << "For " << f.getMetricName() << " : " << f.getFitness(exactQueryFuncationValues.data(), queryFuncationValues.data(), numQueryPoints) << std::endl;
        }

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/RASTRIGIN_TRAIN", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(dataPoints, numDimensions, dataFuncationValues, 1, numDataPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/RASTRIGIN_DCPSE_2", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }

            // Open a file for reading and writing
            if (file.openFile("./dcpse/RASTRIGIN_EXACT", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, exactQueryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }

    {
        // Create an instance of a DC-PSE object
        umuq::dcpse<double> dc(numDimensions);

        // Compute the interpolator weights & operator kernel
        EXPECT_TRUE(dc.computeInterpolatorWeights(dataPoints.data(), numDataPoints, queryPoints.data(), numQueryPoints, 3));

        // Compute the interpolated values
        EXPECT_TRUE(dc.interpolate(dataFuncationValues.data(), numDataPoints, queryFuncationValues.data(), numQueryPoints));

        // Printing the information
        dc.printInfo();

        // Writing to external files
        if (WRITE_TO_FILE)
        {
            // Create an instance of io object
            umuq::io file;

            // Open a file for reading and writing
            if (file.openFile("./dcpse/RASTRIGIN_DCPSE_3", file.in | file.out | file.trunc))
            {
                // Write the matrix in it
                file.saveMatrix<double>(queryPoints, numDimensions, queryFuncationValues, 1, numQueryPoints);
                file.closeFile();
            }
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment<>);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}
