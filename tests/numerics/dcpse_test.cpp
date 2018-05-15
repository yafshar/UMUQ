#include "core/core.hpp"
#include "io/io.hpp"
#include "numerics/dcpse.hpp"
#include "gtest/gtest.h"

/*! 
 * Test to check dcpse functionality
 */
TEST(dcpse_test, HandlesDCoperators)
{
    io f;
    EXPECT_TRUE(f.openFile("numerics/knearestneighbors_test.txt"));
    int nRows = 0;
    while (f.readLine())
    {
        //Count the number of non empty and not commented line with "#" as default comment
        nRows++;
    }

    //This data type has two dimensions
    int nDim = 2;
    //Number of nearest neighbors to find
    int nn = 3;

    double *data = nullptr;
    double *qdata = nullptr;

    int nSize = nRows * nDim;

    //Allocate memory for reading the data
    try
    {
        data = new double[nSize];
        qdata = new double[4];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    //!Rewind the file
    f.rewindFile();

    //!Read the array of data
    EXPECT_TRUE(f.loadMatrix<double>(data, nRows, nDim));

    //Close the file
    f.closeFile();

    qdata[0] = 6.5;
    qdata[1] = 0.5;

    qdata[2] = 6.4;
    qdata[3] = 0.4;

    dcpse<double> dc(nDim);

    EXPECT_TRUE(dc.computeInterpolatorWeights(data, nRows, qdata, 2, 2, 0));

    // EMatrix3d yaser = EMatrix3d::Zero();
    // EVector3d y1, y2, y3;

    // y1 << 1, 2, 3;
    // y2 << 4, 5, 6;
    // y3 << 7, 8, 9;

    // yaser << y1, y2, y3;

    // std::cout << "-----------" << std::endl;
    // std::cout << yaser << std::endl;
    // std::cout << "-----------" << std::endl;

    // double y4[3] = {100, 200, 300};

    // yaser.block(0, 1, 3, 1) << Eigen::Map<EVector3d>(y4);
    // std::cout << yaser << std::endl;
    // std::cout << "-----------" << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
