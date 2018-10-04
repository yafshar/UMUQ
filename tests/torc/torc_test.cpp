#include "core/core.hpp"
#include "environment.hpp"
#include "data/mpidatatype.hpp"
#include "gtest/gtest.h"

/*!
 * \brief This class is designed to test TORC functionality for virtual function
 * 
 */
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

/*!
 * \brief This class is designed to test TORC functionality for virtual function
 * 
 */
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

/*!
 * \brief Function for TORC test
 * 
 * \param other Casted pointer to class A 
 * \param a1 Input value 
 * \param a2 Input value
 */
void FUN(long long const other, int const a1, double const *a2)
{
    auto obj = reinterpret_cast<A *>(other);
    obj->fun(a1, a2);
}

/*!
 * \brief This class is designed to test TORC functionality
 * 
 */
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
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment<>);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}