#include "misc/arraywrapper.hpp"
#include "gtest/gtest.h"

#include <numeric>

/*!
 * \ingroup Test_Module
 *
 * \brief Construct a new TEST object for checking arrayWrapper
 *
 */
TEST(arraywrapper_test, HandlesVectors)
{
    int j;

    // check for array of int
    {
        std::unique_ptr<int[]> iPointer(new int[1000]);

        for (j = 0; j < 1000; j++)
        {
            iPointer[j] = j * 10;
        }

        j = 0;
        umuq::arrayWrapper<int> iArray(iPointer, 1000);
        for (auto i = iArray.begin(); i != iArray.end(); i++, j++)
        {
            EXPECT_EQ(*i, j * 10);
        }
    }

    // check for array of double
    {
        std::unique_ptr<double[]> dPointer(new double[100]);

        for (j = 0; j < 100; j++)
        {
            dPointer[j] = (double)j * 10.0;
        }

        j = 0;
        umuq::arrayWrapper<double> dArray(dPointer, 100);
        for (auto i = dArray.begin(); i != dArray.end(); i++)
        {
            EXPECT_EQ(*i, (double)j * 10);
            j++;
        }
    }

    // check for std::vector of double
    {
        std::vector<double> dPointer(100);

        // Fill the array with some values
        std::iota(dPointer.begin(), dPointer.end(), 100);

        j = 100;
        umuq::arrayWrapper<double> dArray(dPointer);
        for (auto i = dArray.begin(); i != dArray.end(); i++)
        {
            EXPECT_EQ(*i, (double)j);
            j++;
        }
    }

    // check for array of structure
    {
        struct srt
        {
            int i;
            int n;
            double s;
        };

        std::unique_ptr<srt[]> srtPointer(new srt[10]);

        for (j = 0; j < 10; j++)
        {
            srtPointer[j].i = j;
            srtPointer[j].n = j * 10;
            srtPointer[j].s = (double)j * 10.0;
        }

        j = 0;
        umuq::arrayWrapper<srt> srtArray(srtPointer, 10);
        for (auto i = srtArray.begin(); i != srtArray.end(); i++, j++)
        {
            auto e = i.get();

            EXPECT_EQ(e->i, j);
            EXPECT_EQ(e->n, j * 10);
            EXPECT_EQ(e->s, (double)j * 10.0);
        }
    }
}

/*!
 * \ingroup Test_Module
 *
 * \brief Construct a new TEST object for checking arrayWrapper when we have stride
 *
 */
TEST(arraywrapper_test, HandlesVectorsWithStride)
{
    // check for array of int with stride 2
    {
        std::unique_ptr<int[]> iPointer{new int[10]};

        // Fill the array with some values
        std::iota(iPointer.get(), iPointer.get() + 10, 0);

        umuq::arrayWrapper<int> iArray(iPointer, 10, 2);

        int j = 0;
        for (auto i = iArray.begin(); i != iArray.end(); i++)
        {
            EXPECT_EQ(*i, j);
            j += 2;
        }

        EXPECT_EQ(iArray.size(), std::size_t{5});
    }

    // check for array of double with stride 9
    {
        std::unique_ptr<double[]> dPointer{new double[100]};

        // Fill the array with some values
        std::iota(dPointer.get(), dPointer.get() + 100, 1000.);

        umuq::arrayWrapper<double> dArray(dPointer, 100, 9);

        double sd = 1000.;
        for (auto i = dArray.begin(); i != dArray.end(); i++)
        {
            EXPECT_DOUBLE_EQ(*i, sd);
            sd += 9.;
        }

        EXPECT_EQ(dArray.size(), std::size_t{11});
    }

    // check for vector of double with stride 7
    {
        std::vector<double> dPointer(100);

        // Fill the array with some values
        std::iota(dPointer.begin(), dPointer.end(), 1000.);

        umuq::arrayWrapper<double> dArray(dPointer, 7);

        double sd = 1000.;
        for (auto i = dArray.begin(); i != dArray.end(); i++)
        {
            EXPECT_DOUBLE_EQ(*i, sd);
            sd += 7.;
        }

        EXPECT_EQ(dArray.size(), std::size_t{14});
    }
}

/*!
 * \ingroup Test_Module
 *
 * \brief Construct a new TEST object for checking arrayWrapper for N dimensional vectors
 *
 */
TEST(arraywrapper_test, HandlesNDimVectorsWithStride)
{
    // check for 2-Dimensional array
    {
        std::unique_ptr<int[]> iPointer{new int[20]};

        //Fill the array with some values
        for (int i = 0, j = 0; i < 10; i++)
        {
            for (int d = 0; d < 2; d++, j++)
            {
                iPointer[j] = j;
            }
        }

        umuq::arrayWrapper<int> iArray(iPointer, 10, 2);

        {
            int j = 0;
            for (auto i = iArray.begin(); i != iArray.end(); i++)
            {
                EXPECT_EQ(*i, j * 2);
                j++;
            }
        }

        EXPECT_EQ(iArray.size(), std::size_t{5});

        umuq::arrayWrapper<int> jArray(iPointer.get() + 1, 10, 2);

        {
            int j = 0;
            for (auto i = jArray.begin(); i != jArray.end(); i++)
            {
                EXPECT_EQ(*i, j * 2 + 1);
                j++;
            }
        }

        EXPECT_EQ(jArray.size(), std::size_t{5});
    }
}

/*!
 * \ingroup Test_Module
 *
 * \brief Construct a new TEST object for writing at elements
 *
 */
TEST(arraywrapper_test, HandlesWriteInNDimVectorsWithStride)
{
    // check for std::vector of double
    {
        std::vector<double> dPointer(100, 10.);

        umuq::arrayWrapper<double> dArray(dPointer, 10);
        for (auto i : dArray)
        {
            i *= 100;
        }

        int j = 0;
        for (auto i = dPointer.begin(); i != dPointer.end(); i++)
        {
            EXPECT_DOUBLE_EQ(*i, (j % 10 ? 1000. : 10.));
        }
    }

    // check for 2-Dimensional array
    {
        std::unique_ptr<int[]> iPointer{new int[20]};

        // Fill the array with some values
        for (int i = 0, j = 0; i < 10; i++)
        {
            for (int d = 0; d < 2; d++, j++)
            {
                iPointer[j] = j;
            }
        }

        umuq::arrayWrapper<int> iArray(iPointer, 10, 2);

        for (auto i = iArray.begin(); i != iArray.end(); i += 2)
        {
            *i *= 10;
        }

        EXPECT_DOUBLE_EQ(iPointer[4], 40.);
        EXPECT_DOUBLE_EQ(iPointer[8], 80.);
    }
}

/*!
 * \ingroup Test_Module
 *
 * \brief Construct a new TEST object for writing at elements
 *
 */
TEST(arraywrapper_test, HandlesStatisticsFunctions)
{
    // test for normal array of data
    std::vector<int> iVector{2, 3, 5, 7, 1, 6, 8, 10, 9, 4, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10};

    umuq::arrayWrapper<int> iArray(iVector, 2);

    EXPECT_EQ(*std::min_element(iArray.begin(), iArray.end()), -9);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
