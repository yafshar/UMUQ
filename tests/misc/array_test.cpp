#include "core/core.hpp"
#include "misc/array.hpp"
#include "gtest/gtest.h"

TEST(arraywrapper_test, HandlesVectors)
{
    int j;

    //! check for array of int
    int *iarray = nullptr;
    iarray = new int[1000];
    for (j = 0; j < 1000; j++)
    {
        iarray[j] = j * 10;
    }

    j = 0;
    ArrayWrapper<int> it(iarray, 1000);
    for (auto i = it.begin(); i != it.end(); i++, j++)
    {
        EXPECT_EQ(*i, j * 10);
    }

    delete[] iarray;

    //! check for array of double
    double *darray = nullptr;
    darray = new double[100];
    for (j = 0; j < 100; j++)
    {
        darray[j] = (double)j * 10.0;
    }

    j = 0;
    ArrayWrapper<double> id(darray, 100);
    for (auto i = id.begin(); i != id.end(); i++)
    {
        EXPECT_EQ(*i, (double)j * 10);
        j++;
    }

    delete[] darray;

    //! check for array of structure
    struct srt
    {
        int i;
        int n;
        double s;
    };

    srt *srtarray = nullptr;
    srtarray = new srt[10];
    for (j = 0; j < 10; j++)
    {
        srtarray[j].i = j;
        srtarray[j].n = j * 10;
        srtarray[j].s = (double)j * 10.0;
    }

    j = 0;
    ArrayWrapper<srt> is(srtarray, 10);
    for (auto i = is.begin(); i != is.end(); i++, j++)
    {
        auto e = i.get();

        EXPECT_EQ(e->i, j);
        EXPECT_EQ(e->n, j * 10);
        EXPECT_EQ(e->s, (double)j * 10.0);
    }

    delete[] srtarray;
}

TEST(arraywrapper_test, HandlesVectorsWithStride)
{
    //! check for array of int with stride 2
    std::unique_ptr<int[]> ia{new int[10]};

    //Fill the array with some values
    {
        int *iarray = ia.get();
        std::iota(iarray, iarray + 10, 0);
    }

    ArrayWrapper<int> it(ia, 10, 2);

    int j = 0;
    for (auto i = it.begin(); i != it.end(); i++)
    {
        EXPECT_EQ(*i, j);
        j += 2;
    }

    EXPECT_EQ(it.size(), 5);

    //! check for array of double with stride 9
    std::unique_ptr<double[]> da{new double[100]};

    //Fill the array with some values
    {
        double *darray = da.get();
        std::iota(darray, darray + 100, 1000.);
    }

    ArrayWrapper<double> itd(da, 100, 9);

    double sd = 1000.;
    for (auto i = itd.begin(); i != itd.end(); i++)
    {
        EXPECT_DOUBLE_EQ(*i, sd);
        sd += 9.;
    }

    EXPECT_EQ(itd.size(), 11);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}