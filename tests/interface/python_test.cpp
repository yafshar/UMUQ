#include "interface/python.hpp"
#include "environment.hpp"
#include "gtest/gtest.h"

#ifdef HAVE_PYTHON

#include <cstring>
#include <vector>
#include <numeric>

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

// Tests python
TEST(numpy_test, HandlesConstruction)
{
    using namespace umuq::python;
    using namespace umuq::python::numpy;

    {
        std::vector<int> a(10);
        std::iota(a.begin(), a.end(), 0);
        auto Obj1 = PyArray<int>(a);
        EXPECT_TRUE((Obj1));

        auto Obj2 = PyArray<int, double>(a);
        EXPECT_TRUE((Obj2));
    }

    {
        std::vector<double> a(10);
        std::iota(a.begin(), a.end(), 0.0);
        auto Obj1 = PyArray<double>(a);
        EXPECT_TRUE((Obj1));

        auto Obj2 = PyArray<double, int>(a);
        EXPECT_TRUE((Obj2));
    }

    {
        auto Obj1 = PyArray<double>(1., 100);
        EXPECT_TRUE((Obj1));

        auto Obj2 = PyArray<int>(0, 10);
        EXPECT_TRUE((Obj2));

        auto Obj3 = PyArray<char>('r', 10);
        EXPECT_TRUE((Obj3));
    }

    {
        std::vector<int> a(100);
        std::iota(a.begin(), a.end(), 0);

        auto Obj1 = PyArray<int>(a.data(), 100, 10);
        EXPECT_TRUE((Obj1));

        auto Obj2 = PyArray<int, double>(a.data(), 100, 10);
        EXPECT_TRUE((Obj2));
    }

    {
        std::vector<int> a(12);
        std::iota(a.begin(), a.end(), 0);
        auto Obj1 = Py2DArray<int>(a, 3, 4);
        EXPECT_TRUE((Obj1));

        auto Obj2 = Py2DArray<int, double>(a, 3, 4);
        EXPECT_TRUE((Obj2));
    }

    {
        std::vector<int> a(12);
        std::iota(a.begin(), a.end(), 0);
        auto Obj1 = Py2DArray<int>(a.data(), 3, 4);
        EXPECT_TRUE((Obj1));

        auto Obj2 = Py2DArray<int, double>(a.data(), 3, 4);
        EXPECT_TRUE((Obj2));
    }
}

// Tests python calling function name (statistics function)
TEST(python_test, HandlesCallFunctionName)
{
    using namespace umuq::python;

    {
        std::vector<int> a(10);
        std::iota(a.begin(), a.end(), 0);
        auto Obj1 = PyArray<int>(a);
        EXPECT_TRUE((Obj1));

        std::string functionName = "statistics.mean";
        auto Obj2 = PyCallFunctionName(functionName, Obj1);
        EXPECT_TRUE((Obj2));

        EXPECT_DOUBLE_EQ(PyFloat_AS_DOUBLE(Obj2), 4.5);
    }

    {
        std::vector<int> a(10);
        std::iota(a.begin(), a.end(), 0);
        auto Obj1 = PyArray<int>(a);
        EXPECT_TRUE((Obj1));

        std::string functionName = "statistics.mean";
        auto Obj2 = PyCallFunctionName(functionName, PyObjectVector{Obj1});
        EXPECT_TRUE((Obj2));

        EXPECT_DOUBLE_EQ(PyFloat_AS_DOUBLE(Obj2), 4.5);
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
