#include "core/core.hpp"
#include "environment.hpp"
#include "data/database.hpp"
#include "gtest/gtest.h"

//! Tests databse constrcution
TEST(database_test, HandlesConstruction)
{
    //! Create an instance of database object
    {
        database<double> d;

        EXPECT_EQ(0, d.ndimParray);
        EXPECT_EQ(0, d.ndimGarray);
        EXPECT_EQ(std::size_t{}, d.idxPos);
        EXPECT_EQ(std::size_t{}, d.entries);
    }

    std::remove("database_100.txt");

    //! Create an instance of database object
    {
        database<double> d(2, 2, 3);

        EXPECT_EQ(2, d.ndimParray);
        EXPECT_EQ(2, d.ndimGarray);
        EXPECT_EQ(std::size_t{}, d.idxPos);
        EXPECT_EQ(std::size_t{3}, d.entries);

        {
            double p[] = {1., -1.};
            double g[] = {120., 321.};
            d.update(p, 1000., g, 1);
        }

        {
            double p[] = {2, 3.4};
            double g[] = {1206., 3621.};
            d.update(p, 10000., g, 1);
        }

        {
            double p[] = {4., 14};
            double g[] = {506., 132621.};
            d.update(p, 2000, g, 0);
        }

        EXPECT_DOUBLE_EQ(1., d.Parray[0]);
        EXPECT_DOUBLE_EQ(-1., d.Parray[1]);
        EXPECT_DOUBLE_EQ(2., d.Parray[2]);
        EXPECT_DOUBLE_EQ(3.4, d.Parray[3]);
        EXPECT_DOUBLE_EQ(4., d.Parray[4]);
        EXPECT_DOUBLE_EQ(14., d.Parray[5]);

        EXPECT_DOUBLE_EQ(120., d.Garray[0]);
        EXPECT_DOUBLE_EQ(321., d.Garray[1]);
        EXPECT_DOUBLE_EQ(1206., d.Garray[2]);
        EXPECT_DOUBLE_EQ(3621, d.Garray[3]);
        EXPECT_DOUBLE_EQ(506., d.Garray[4]);
        EXPECT_DOUBLE_EQ(132621., d.Garray[5]);

        EXPECT_DOUBLE_EQ(1000., d.Fvalue[0]);
        EXPECT_DOUBLE_EQ(10000., d.Fvalue[1]);
        EXPECT_DOUBLE_EQ(2000., d.Fvalue[2]);

        EXPECT_EQ(1, d.Surrogate[0]);
        EXPECT_EQ(1, d.Surrogate[1]);
        EXPECT_EQ(0, d.Surrogate[2]);

        EXPECT_TRUE(d.save("database", 100));
    }


    {
        database<double> d;
        EXPECT_FALSE(d.load("database", 100));
    }

    {
        database<double> d(2, 2, 3);
        EXPECT_TRUE(d.load("database", 100));

        EXPECT_DOUBLE_EQ(1., d.Parray[0]);
        EXPECT_DOUBLE_EQ(-1., d.Parray[1]);
        EXPECT_DOUBLE_EQ(2., d.Parray[2]);
        EXPECT_DOUBLE_EQ(3.4, d.Parray[3]);
        EXPECT_DOUBLE_EQ(4., d.Parray[4]);
        EXPECT_DOUBLE_EQ(14., d.Parray[5]);

        EXPECT_DOUBLE_EQ(120., d.Garray[0]);
        EXPECT_DOUBLE_EQ(321., d.Garray[1]);
        EXPECT_DOUBLE_EQ(1206., d.Garray[2]);
        EXPECT_DOUBLE_EQ(3621, d.Garray[3]);
        EXPECT_DOUBLE_EQ(506., d.Garray[4]);
        EXPECT_DOUBLE_EQ(132621., d.Garray[5]);

        EXPECT_DOUBLE_EQ(1000., d.Fvalue[0]);
        EXPECT_DOUBLE_EQ(10000., d.Fvalue[1]);
        EXPECT_DOUBLE_EQ(2000., d.Fvalue[2]);
    }

    std::remove("database_100.txt");
}

//! Tests datatype which is using database object
TEST(database_test, HandlesTask)
{
    {
        //Create an instance of a database object
        database<double> d(2, 2, 3);

        //Update the data using different threads
        {
            double p[] = {1., -1.};
            double g[] = {120., 321.};
            d.update(p, 1000., g, 1);
        }

        {
            double p[] = {2, 3.4};
            double g[] = {1206., 3621.};
            d.update(p, 10000., g, 1);
        }

        {
            double p[] = {4., 14};
            double g[] = {506., 132621.};
            d.update(p, 2000, g, 0);
        }

        EXPECT_DOUBLE_EQ(1., d.Parray[0]);
        EXPECT_DOUBLE_EQ(-1., d.Parray[1]);
        EXPECT_DOUBLE_EQ(2., d.Parray[2]);
        EXPECT_DOUBLE_EQ(3.4, d.Parray[3]);
        EXPECT_DOUBLE_EQ(4., d.Parray[4]);
        EXPECT_DOUBLE_EQ(14., d.Parray[5]);

        EXPECT_DOUBLE_EQ(120., d.Garray[0]);
        EXPECT_DOUBLE_EQ(321., d.Garray[1]);
        EXPECT_DOUBLE_EQ(1206., d.Garray[2]);
        EXPECT_DOUBLE_EQ(3621, d.Garray[3]);
        EXPECT_DOUBLE_EQ(506., d.Garray[4]);
        EXPECT_DOUBLE_EQ(132621., d.Garray[5]);

        EXPECT_DOUBLE_EQ(1000., d.Fvalue[0]);
        EXPECT_DOUBLE_EQ(10000., d.Fvalue[1]);
        EXPECT_DOUBLE_EQ(2000., d.Fvalue[2]);

        EXPECT_EQ(1, d.Surrogate[0]);
        EXPECT_EQ(1, d.Surrogate[1]);
        EXPECT_EQ(0, d.Surrogate[2]);

        EXPECT_TRUE(d.save("database", 100));
    }

    std::remove("database_100.txt");
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