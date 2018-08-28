#include "core/core.hpp"
#include "core/environment.hpp"
#include "data/mpidatatype.hpp"
#include "gtest/gtest.h"

class A
{
  public:
    A() : a(10) {}
    int a;
    virtual void fun(int const a1, double const *a2)
    {
        std::cout << std::endl;
        std::cout << "Class A, a1=" << a1 << ", a2=" << *a2 << std::endl;
        std::cout << std::endl;
    }
};

class B : public A
{
  public:
    B() : A(), aa(100), aaa(-1.) {}
    int aa;
    double aaa;
    void fun(int const a1, double const *a2)
    {
        std::cout << std::endl;
        std::cout << "Class B, a1=" << a1 << ", a2=" << *a2 << std::endl;
        std::cout << std::endl;
    }
};

void FUN(long long const other, int const a1, double const *a2)
{
    auto obj = reinterpret_cast<A *>(other);
    obj->fun(a1, a2);
}

class C
{
  public:
    void loop()
    {
        A obj1;
        B obj2;

        int a1 = 100;
        double a2[] = {-10., 20.};

        for (int i = 0; i < 4; i++)
        {
            a1 = 100;
            a2[0] *= (i + 1);
            torc_create(-1, (void (*)())FUN, 3,
                        1, MPIDatatype<long long>, CALL_BY_REF,
                        1, MPI_INT, CALL_BY_REF,
                        1, MPI_DOUBLE, CALL_BY_VAL,
                        reinterpret_cast<long long>(&obj1), a1, a2);

            a1 = 10000;
            a2[0] *= (i + 2);
            torc_create(-1, (void (*)())FUN, 3,
                        1, MPIDatatype<long long>, CALL_BY_REF,
                        1, MPI_INT, CALL_BY_REF,
                        1, MPI_DOUBLE, CALL_BY_VAL,
                        reinterpret_cast<long long>(&obj2), a1, a2);
        }

        torc_waitall();
    }
};

//! Tests torc
TEST(torc_test, HandlesClass)
{
    torc_register_task((void *)FUN);

    C obj;
    obj.loop();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}