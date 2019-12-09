#include "interface/python.hpp"
#include "environment.hpp"
#include "gtest/gtest.h"

#ifdef HAVE_PYTHON

#include <cstring>

// Tests python
TEST(python_test, HandlesConstruction)
{
    using namespace umuq::python;
    {
        int const a = 100;
        auto Obj1 = PyObjectConstruct(a);
        auto Obj2 = PyObjectConstruct(a);
        EXPECT_TRUE(Obj1 == Obj2);
        EXPECT_TRUE(static_cast<int>(PyLong_AsLong(Obj1)) == a);
    }
    {
        int const a = 100;
        int const b = 101;
        auto Obj1 = PyObjectConstruct(a);
        auto Obj2 = PyObjectConstruct(b);
        EXPECT_TRUE(Obj1 != Obj2);
        EXPECT_TRUE(static_cast<int>(PyLong_AsLong(Obj1)) == a);
        EXPECT_TRUE(static_cast<int>(PyLong_AsLong(Obj2)) == b);
        EXPECT_TRUE(PyLong_AsLong(Obj1) < PyLong_AsLong(Obj2));
    }
    {
        float const a = 100.;
        int const b = 100;
        auto Obj1 = PyObjectConstruct(a);
        auto Obj2 = PyObjectConstruct(b);
        EXPECT_TRUE(Obj1 != Obj2);
    }
    {
        double const a = 100.;
        double const b = 100.;
        auto Obj1 = PyObjectConstruct(a);
        auto Obj2 = PyObjectConstruct(b);
        EXPECT_DOUBLE_EQ(PyFloat_AsDouble(Obj1), PyFloat_AsDouble(Obj2));
    }
    {
        std::string const a = "This is a python module";
        auto Obj1 = PyObjectConstruct(a);
        EXPECT_TRUE(std::strcmp(PyString_AsString(Obj1), a.c_str()) == 0);
    }
    {
        std::vector<int> a(10, 1);
        auto Obj1 = PyObjectConstruct<int>(a);
        EXPECT_TRUE((Obj1));

        std::vector<unsigned int> b(100, 2);
        auto Obj2 = PyObjectConstruct<unsigned int>(b);
        EXPECT_FALSE((Obj2));
    }
}
#else
// Tests python
TEST(python_test, HandlesConstruction)
{
}
#endif // HAVE_PYTHON

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}
