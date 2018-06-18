#include "core/core.hpp"
#include "core/environment.hpp"
#include "data/datatype.hpp"
#include "gtest/gtest.h"

//! Tests datatype which is using database object
TEST(datatype_test, HandlesGlobalData)
{
    {
        //Create an instance of a database object
        database<double> d(2, 2, 3);

        //Assign (Move assignment) Data1 to the constructed object
        Data1<double> = std::move(d);
    }
    {
        //Make a pointer to the Data1 for ease of use
        auto *d = &Data1<double>;

        //Set the update Task function to be used for updating on multi threads or processors
        d->set(updateTask1<double>);

        //Initilize the update Task 
        EXPECT_TRUE(d->init());

        //Update the data using different threads
        {
            double p[] = {1., -1.};
            double g[] = {120., 321.};
            d->update(p, 1000., g, 1);
        }

        {
            double p[] = {2, 3.4};
            double g[] = {1206., 3621.};
            d->update(p, 10000., g, 1);
        }

        {
            double p[] = {4., 14};
            double g[] = {506., 132621.};
            d->update(p, 2000, g, 0);
        }

        EXPECT_DOUBLE_EQ(1., d->Parray[0]);
        EXPECT_DOUBLE_EQ(-1., d->Parray[1]);
        EXPECT_DOUBLE_EQ(2., d->Parray[2]);
        EXPECT_DOUBLE_EQ(3.4, d->Parray[3]);
        EXPECT_DOUBLE_EQ(4., d->Parray[4]);
        EXPECT_DOUBLE_EQ(14., d->Parray[5]);

        EXPECT_DOUBLE_EQ(120., d->Garray[0]);
        EXPECT_DOUBLE_EQ(321., d->Garray[1]);
        EXPECT_DOUBLE_EQ(1206., d->Garray[2]);
        EXPECT_DOUBLE_EQ(3621, d->Garray[3]);
        EXPECT_DOUBLE_EQ(506., d->Garray[4]);
        EXPECT_DOUBLE_EQ(132621., d->Garray[5]);

        EXPECT_DOUBLE_EQ(1000., d->Fvalue[0]);
        EXPECT_DOUBLE_EQ(10000., d->Fvalue[1]);
        EXPECT_DOUBLE_EQ(2000., d->Fvalue[2]);

        EXPECT_EQ(1, d->Surrogate[0]);
        EXPECT_EQ(1, d->Surrogate[1]);
        EXPECT_EQ(0, d->Surrogate[2]);

        EXPECT_TRUE(d->save("database", 100));
    }

    std::remove("database_100.txt");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}