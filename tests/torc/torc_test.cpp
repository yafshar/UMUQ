#include "core/core.hpp"
#include "environment.hpp"
#include "datatype/mpidatatype.hpp"
#include "gtest/gtest.h"

/*! \class baseA
 * \ingroup Test_Module
 * 
 * \brief This class is designed to test TORC functionality for virtual function
 * 
 */
class baseA
{
  public:
    baseA() : a(10) {}
    int a;
    virtual void fun(int const a1, double const *a2)
    {
        std::cout << std::endl;
        std::cout << "Class baseA, a1=" << a1 << ", a2=" << *a2 << std::endl;
        std::cout << std::endl;
    }
};

/*! \class driveA
 * \ingroup Test_Module
 * 
 * \brief This class is designed to test TORC functionality for virtual function
 * 
 */
class driveA : public baseA
{
  public:
    driveA() : baseA(), aa(100), aaa(-1.) {}
    int aa;
    double aaa;
    void fun(int const a1, double const *a2)
    {
        std::cout << std::endl;
        std::cout << "Class driveA, a1=" << a1 << ", a2=" << *a2 << std::endl;
        std::cout << std::endl;
    }
};

/*!
 * \ingroup Test_Module
 * 
 * \brief Function for TORC test
 * 
 * \param other Casted pointer to class baseA 
 * \param a1 Input value 
 * \param a2 Input value
 */
void FUN(long long const other, int const a1, double const *a2)
{
    auto obj = reinterpret_cast<baseA *>(other);
    obj->fun(a1, a2);
}

/*! \class torcTest
 * \ingroup Test_Module
 * 
 * \brief This class is designed to test TORC functionality
 * 
 */
class torcTest
{
  public:
    void loop()
    {
        baseA obj1;
        driveA obj2;

        int a1 = 100;
        double a2[] = {-10., 20.};

        for (int i = 0; i < 4; i++)
        {
            a1 = 100;
            a2[0] *= (i + 1);
            torc_create(-1, (void (*)())FUN, 3,
                        1, umuq::MPIDatatype<long long>, CALL_BY_REF,
                        1, MPI_INT, CALL_BY_REF,
                        1, MPI_DOUBLE, CALL_BY_VAL,
                        reinterpret_cast<long long>(&obj1), a1, a2);

            a1 = 10000;
            a2[0] *= (i + 2);
            torc_create(-1, (void (*)())FUN, 3,
                        1, umuq::MPIDatatype<long long>, CALL_BY_REF,
                        1, MPI_INT, CALL_BY_REF,
                        1, MPI_DOUBLE, CALL_BY_VAL,
                        reinterpret_cast<long long>(&obj2), a1, a2);
        }

        torc_waitall();
    }
};

/*! Tests torc */
TEST(torc_test, HandlesClass)
{
    torc_register_task((void *)FUN);

    torcTest obj;
    obj.loop();
}

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