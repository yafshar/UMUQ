#include "core/core.hpp"
#include "numerics/multimin.hpp"
#include "numerics/testfunctions/optimizationtestfunctions.hpp"
#include "gtest/gtest.h"

using namespace multimin;

template <typename T, class TMFDMT, class TMFD>
bool test_fdf(const char *desc, TMFDMT *ftype, TMFD *f)
{
    std::size_t n = f->n;

    T *x = new T[n];

    f->initpt(x);

    T step_size;
    {
        T sum(0);
        std::for_each(x, x + n, [&](T const x_i) { sum += x_i * x_i; });
        step_size = 0.1 * std::sqrt(sum);
    }

    //create an instance of object for minimizing functions using derivatives
    multimin_fdfminimizer<T, TMFDMT, TMFD> mfdf_obj;

    mfdf_obj.alloc(ftype, n);

    mfdf_obj.set(f, x, step_size, 0.1);

#ifdef DEBUG
    {
        T const *x_t = mfdf_obj.get_x();

        std::cout << "x =";
        for (std::size_t i = 0; i < n; i++)
        {
            std::cout << x_t[i] << " ";
        }
        std::cout << std::endl;

        T const *g_t = mfdf_obj.get_gradient();
        std::cout << "g =";
        for (std::size_t i = 0; i < n; i++)
        {
            std::cout << g_t[i] << " ";
        }
        std::cout << std::endl;
    }
#endif

    //fail    -1
    //success  0
    //continue 1
    int status = 1;

    std::size_t iter = 0;

    while (iter < 5000 && status == 1)
    {
        iter++;
        mfdf_obj.iterate();

#ifdef DEBUG
        {
            std::cout << iter << ": " << std::endl;

            T const *x_t = mfdf_obj.get_x();

            std::cout << "x ";
            for (std::size_t i = 0; i < n; i++)
            {
                std::cout << x_t[i] << " ";
            }
            std::cout << std::endl;

            T const *g_t = mfdf_obj.get_gradient();

            std::cout << "g =";
            for (std::size_t i = 0; i < n; i++)
            {
                std::cout << g_t[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "f(x) =" << mfdf_obj.minimum() << std::endl;

            T const *dx = mfdf_obj.get_dx();
            T sum(0);
            std::for_each(dx, dx + n, [&](T const d_i) { sum += d_i * d_i; });
            std::cout << "dx =" << std::sqrt(sum) << std::endl;

            std::cout << std::endl;
        }
#endif

        T const *gradient = mfdf_obj.get_gradient();

        status = multimin_test_gradient<T>(gradient, n, 1e-3);
    }

    std::cout << mfdf_obj.name() << ", on " << desc << ": " << iter << " iters, f(x)=" << mfdf_obj.minimum() << std::endl;

    mfdf_obj.free();
    delete[] x;

    return ((status == 0) ? true : (status == 1) ? (std::abs(mfdf_obj.minimum()) > 1e-5) : false);
}

template <typename T, class TMFMT, class TMF>
bool test_f(const char *desc, TMFMT *ftype, TMF *f)
{
    std::size_t n = f->n;

    T *x = new T[n];

    f->initpt(x);

    T *step_size = new T[n];

    multimin_fminimizer<T, TMFMT, TMF> mff_obj;

    mff_obj.alloc(ftype, n);

    for (std::size_t i = 0; i < n; i++)
    {
        step_size[i] = 1;
    }

    mff_obj.set(f, x, step_size);

#ifdef DEBUG
    T const *x_t = mff_obj.get_x();

    std::cout << "x =";
    for (std::size_t i = 0; i < n; i++)
    {
        std::cout << x_t[i] << " ";
    }
    std::cout << std::endl;
#endif

    //fail    -1
    //success  0
    //continue 1
    int status = 1;

    std::size_t iter = 0;

    while (iter < 5000 && status == 1)
    {
        iter++;
        mff_obj.iterate();

#ifdef DEBUG
        {
            std::cout << iter << ": " << std::endl;

            T const *x_t = mff_obj.get_x();

            std::cout << "x ";
            for (std::size_t i = 0; i < n; i++)
            {
                std::cout << x_t[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "f(x) =" << mff_obj.minimum() << std::endl;
            std::cout << "size: " << mff_obj.get_size() << std::endl;
            std::cout << std::endl;
        }
#endif

        status = multimin_test_size<T>(mff_obj.get_size(), 1e-3);
    }

    std::cout << mff_obj.name() << ", on " << desc << ": " << iter << " iters, f(x)=" << mff_obj.minimum() << std::endl;

    mff_obj.free();
    delete[] x;
    delete[] step_size;

    return ((status == 0) ? true : (status == 1) ? (std::abs(mff_obj.minimum()) > 1e-5) : false);
}

typedef rosenbrock<double> Rosenbrock;
typedef Nrosenbrock<double> NRosenbrock;
typedef roth<double> Roth;
typedef Nroth<double> NRoth;
typedef wood<double> Wood;
typedef Nwood<double> NWood;

/*!
 * Test to check multimin functionality
 */
TEST(multimin_test_steepest_descent, Handles_Rosenbrock_Function)
{
    typedef steepest_descent<double, Rosenbrock> TRosenbrock;
    typedef steepest_descent<double, NRosenbrock> TNRosenbrock;

    TRosenbrock t;
    Rosenbrock r;

    TNRosenbrock tn;
    NRosenbrock rn;

    EXPECT_TRUE((test_fdf<double, TRosenbrock, Rosenbrock>("Rosenbrock", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRosenbrock, NRosenbrock>("NRosenbrock", &tn, &rn)));
}

TEST(multimin_test_steepest_descent, Handles_Roth_Function)
{
    typedef steepest_descent<double, Roth> TRoth;
    typedef steepest_descent<double, NRoth> TNRoth;

    TRoth t;
    Roth r;

    TNRoth tn;
    NRoth rn;

    EXPECT_TRUE((test_fdf<double, TRoth, Roth>("Roth", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRoth, NRoth>("NRoth", &tn, &rn)));
}

TEST(multimin_test_steepest_descent, Handles_Wood_Function)
{
    typedef steepest_descent<double, Wood> TWood;
    typedef steepest_descent<double, NWood> TNWood;

    TWood t;
    Wood r;

    TNWood tn;
    NWood rn;

    EXPECT_TRUE((test_fdf<double, TWood, Wood>("Wood", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNWood, NWood>("NWood", &tn, &rn)));
}

TEST(multimin_test_conjugate_pr, Handles_Rosenbrock_Function)
{
    typedef conjugate_pr<double, Rosenbrock> TRosenbrock;
    typedef conjugate_pr<double, NRosenbrock> TNRosenbrock;

    TRosenbrock t;
    Rosenbrock r;

    TNRosenbrock tn;
    NRosenbrock rn;

    EXPECT_TRUE((test_fdf<double, TRosenbrock, Rosenbrock>("Rosenbrock", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRosenbrock, NRosenbrock>("NRosenbrock", &tn, &rn)));
}

TEST(multimin_test_conjugate_pr, Handles_Roth_Function)
{
    typedef conjugate_pr<double, Roth> TRoth;
    typedef conjugate_pr<double, NRoth> TNRoth;

    TRoth t;
    Roth r;

    TNRoth tn;
    NRoth rn;

    EXPECT_TRUE((test_fdf<double, TRoth, Roth>("Roth", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRoth, NRoth>("NRoth", &tn, &rn)));
}

TEST(multimin_test_conjugate_pr, Handles_Wood_Function)
{
    typedef conjugate_pr<double, Wood> TWood;
    typedef conjugate_pr<double, NWood> TNWood;

    TWood t;
    Wood r;

    TNWood tn;
    NWood rn;

    EXPECT_TRUE((test_fdf<double, TWood, Wood>("Wood", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNWood, NWood>("NWood", &tn, &rn)));
}

TEST(multimin_test_conjugate_fr, Handles_Rosenbrock_Function)
{
    typedef conjugate_fr<double, Rosenbrock> TRosenbrock;
    typedef conjugate_fr<double, NRosenbrock> TNRosenbrock;

    TRosenbrock t;
    Rosenbrock r;

    TNRosenbrock tn;
    NRosenbrock rn;

    EXPECT_TRUE((test_fdf<double, TRosenbrock, Rosenbrock>("Rosenbrock", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRosenbrock, NRosenbrock>("NRosenbrock", &tn, &rn)));
}

TEST(multimin_test_conjugate_fr, Handles_Roth_Function)
{
    typedef conjugate_fr<double, Roth> TRoth;
    typedef conjugate_fr<double, NRoth> TNRoth;

    TRoth t;
    Roth r;

    TNRoth tn;
    NRoth rn;

    EXPECT_TRUE((test_fdf<double, TRoth, Roth>("Roth", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRoth, NRoth>("NRoth", &tn, &rn)));
}

TEST(multimin_test_conjugate_fr, Handles_Wood_Function)
{
    typedef conjugate_fr<double, Wood> TWood;
    typedef conjugate_fr<double, NWood> TNWood;

    TWood t;
    Wood r;

    TNWood tn;
    NWood rn;

    EXPECT_TRUE((test_fdf<double, TWood, Wood>("Wood", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNWood, NWood>("NWood", &tn, &rn)));
}

TEST(multimin_test_vector_bfgs, Handles_Rosenbrock_Function)
{
    typedef vector_bfgs<double, Rosenbrock> TRosenbrock;
    typedef vector_bfgs<double, NRosenbrock> TNRosenbrock;

    TRosenbrock t;
    Rosenbrock r;

    TNRosenbrock tn;
    NRosenbrock rn;

    EXPECT_TRUE((test_fdf<double, TRosenbrock, Rosenbrock>("Rosenbrock", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRosenbrock, NRosenbrock>("NRosenbrock", &tn, &rn)));
}

TEST(multimin_test_vector_bfgs, Handles_Roth_Function)
{
    typedef vector_bfgs<double, Roth> TRoth;
    typedef vector_bfgs<double, NRoth> TNRoth;

    TRoth t;
    Roth r;

    TNRoth tn;
    NRoth rn;

    EXPECT_TRUE((test_fdf<double, TRoth, Roth>("Roth", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRoth, NRoth>("NRoth", &tn, &rn)));
}

TEST(multimin_test_vector_bfgs, Handles_Wood_Function)
{
    typedef vector_bfgs<double, Wood> TWood;
    typedef vector_bfgs<double, NWood> TNWood;

    TWood t;
    Wood r;

    TNWood tn;
    NWood rn;

    EXPECT_TRUE((test_fdf<double, TWood, Wood>("Wood", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNWood, NWood>("NWood", &tn, &rn)));
}

TEST(multimin_test_vector_bfgs2, Handles_Rosenbrock_Function)
{
    typedef vector_bfgs2<double, Rosenbrock> TRosenbrock;
    typedef vector_bfgs2<double, NRosenbrock> TNRosenbrock;

    TRosenbrock t;
    Rosenbrock r;

    TNRosenbrock tn;
    NRosenbrock rn;

    EXPECT_TRUE((test_fdf<double, TRosenbrock, Rosenbrock>("Rosenbrock", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRosenbrock, NRosenbrock>("NRosenbrock", &tn, &rn)));
}

TEST(multimin_test_vector_bfgs2, Handles_Roth_Function)
{
    typedef vector_bfgs2<double, Roth> TRoth;
    typedef vector_bfgs2<double, NRoth> TNRoth;

    TRoth t;
    Roth r;

    TNRoth tn;
    NRoth rn;

    EXPECT_TRUE((test_fdf<double, TRoth, Roth>("Roth", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNRoth, NRoth>("NRoth", &tn, &rn)));
}

TEST(multimin_test_vector_bfgs2, Handles_Wood_Function)
{
    typedef vector_bfgs2<double, Wood> TWood;
    typedef vector_bfgs2<double, NWood> TNWood;

    TWood t;
    Wood r;

    TNWood tn;
    NWood rn;

    EXPECT_TRUE((test_fdf<double, TWood, Wood>("Wood", &t, &r)));
    EXPECT_TRUE((test_fdf<double, TNWood, NWood>("NWood", &tn, &rn)));
}

typedef rosenbrock_fmin<double> Rosenbrock_f;
typedef roth_fmin<double> Roth_f;
typedef wood_fmin<double> Wood_f;
typedef spring_fmin<double> Spring_f;

TEST(multimin_test_nmsimplex, Handles_Rosenbrock_Function)
{
    typedef nmsimplex<double, Rosenbrock_f> TRosenbrock_f;

    TRosenbrock_f t;
    Rosenbrock_f r;

    EXPECT_TRUE((test_f<double, TRosenbrock_f, Rosenbrock_f>("Rosenbrock", &t, &r)));
}

TEST(multimin_test_nmsimplex, Handles_Roth_Function)
{
    typedef nmsimplex<double, Roth_f> TRoth_f;

    TRoth_f t;
    Roth_f r;

    EXPECT_TRUE((test_f<double, TRoth_f, Roth_f>("Roth", &t, &r)));
}

TEST(multimin_test_nmsimplex, Handles_Wood_Function)
{
    typedef nmsimplex<double, Wood_f> TWood_f;

    TWood_f t;
    Wood_f r;

    EXPECT_TRUE((test_f<double, TWood_f, Wood_f>("Wood", &t, &r)));
}

TEST(multimin_test_nmsimplex, Handles_Spring_Function)
{
    typedef nmsimplex<double, Spring_f> TSpring_f;

    TSpring_f t;
    Spring_f r;

    EXPECT_TRUE((test_f<double, TSpring_f, Spring_f>("Spring", &t, &r)));
}

TEST(multimin_test_nmsimplex2, Handles_Rosenbrock_Function)
{
    typedef nmsimplex2<double, Rosenbrock_f> TRosenbrock_f;

    TRosenbrock_f t;
    Rosenbrock_f r;

    EXPECT_TRUE((test_f<double, TRosenbrock_f, Rosenbrock_f>("Rosenbrock", &t, &r)));
}

TEST(multimin_test_nmsimplex2, Handles_Roth_Function)
{
    typedef nmsimplex2<double, Roth_f> TRoth_f;

    TRoth_f t;
    Roth_f r;

    EXPECT_TRUE((test_f<double, TRoth_f, Roth_f>("Roth", &t, &r)));
}

TEST(multimin_test_nmsimplex2, Handles_Wood_Function)
{
    typedef nmsimplex2<double, Wood_f> TWood_f;

    TWood_f t;
    Wood_f r;

    EXPECT_TRUE((test_f<double, TWood_f, Wood_f>("Wood", &t, &r)));
}

TEST(multimin_test_nmsimplex2, Handles_Spring_Function)
{
    typedef nmsimplex2<double, Spring_f> TSpring_f;

    TSpring_f t;
    Spring_f r;

    EXPECT_TRUE((test_f<double, TSpring_f, Spring_f>("Spring", &t, &r)));
}

TEST(multimin_test_nmsimplex2rand, Handles_Rosenbrock_Function)
{
    typedef nmsimplex2rand<double, Rosenbrock_f> TRosenbrock_f;

    TRosenbrock_f t;
    Rosenbrock_f r;

    EXPECT_TRUE((test_f<double, TRosenbrock_f, Rosenbrock_f>("Rosenbrock", &t, &r)));
}

TEST(multimin_test_nmsimplex2rand, Handles_Roth_Function)
{
    typedef nmsimplex2rand<double, Roth_f> TRoth_f;

    TRoth_f t;
    Roth_f r;

    EXPECT_TRUE((test_f<double, TRoth_f, Roth_f>("Roth", &t, &r)));
}

TEST(multimin_test_nmsimplex2rand, Handles_Wood_Function)
{
    typedef nmsimplex2rand<double, Wood_f> TWood_f;

    TWood_f t;
    Wood_f r;

    EXPECT_TRUE((test_f<double, TWood_f, Wood_f>("Wood", &t, &r)));
}

TEST(multimin_test_nmsimplex2rand, Handles_Spring_Function)
{
    typedef nmsimplex2rand<double, Spring_f> TSpring_f;

    TSpring_f t;
    Spring_f r;

    EXPECT_TRUE((test_f<double, TSpring_f, Spring_f>("Spring", &t, &r)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
