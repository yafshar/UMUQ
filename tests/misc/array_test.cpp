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

        EXPECT_EQ(e.i, j);
        EXPECT_EQ(e.n, j * 10);
        EXPECT_EQ(e.s, (double)j * 10.0);
    }

    delete[] srtarray;
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}