#include "core/core.hpp"
#include "gtest/gtest.h"

#include "numerics/multimin.hpp"
#include "numerics/multimin/multimin_steepest_descent.hpp"
#include "numerics/multimin/multimin_conjugate_fr.hpp"
#include "numerics/multimin/multimin_conjugate_pr.hpp"
#include "numerics/multimin/multimin_vector_bfgs.hpp"
#include "numerics/multimin/multimin_vector_bfgs2.hpp"
#include "numerics/multimin/multimin_nmsimplex.hpp"
#include "numerics/multimin/multimin_nmsimplex2.hpp"
#include "numerics/multimin/multimin_nmsimplex2rand.hpp"

template <typename T>
class rosenbrock_fmin : public multimin_function<T, rosenbrock_fmin<T>>
{
  public:
    rosenbrock_fmin() { this->n = 2; }

    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = u - 1;
        T b = u * u - v;
        return a * a + 10 * b * b;
    }

    void initpt(T *x)
    {
        x[0] = -1.2;
        x[1] = 1.0;
    }
};

template <typename T>
class rosenbrock : public multimin_function_fdf<T, rosenbrock<T>>
{
  public:
    rosenbrock() { this->n = 2; }

    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = u - 1;
        T b = u * u - v;
        return a * a + 10 * b * b;
    }

    T df(T const *x, T *df_)
    {
        T u = x[0];
        T v = x[1];
        T b = u * u - v;

        df_[0] = 2 * (u - 1) + 40 * u * b;
        df_[1] = -20 * b;
    }

    void fdf(T const *x, T *f_, T *df_)
    {
        T u = x[0];
        T v = x[1];
        T a = u - 1;
        T b = u * u - v;
        *f_ = a * a + 10 * b * b;
        df_[0] = 2 * (u - 1) + 40 * u * b;
        df_[1] = -20 * b;
    }

    void initpt(T *x)
    {
        x[0] = -1.2;
        x[1] = 1.0;
    }
};

template <typename T>
class Nrosenbrock : public multimin_function_fdf<T, Nrosenbrock<T>>
{
  public:
    Nrosenbrock() { this->n = 2; }

    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = u - 1;
        T b = u * u - v;
        return a * a + 10 * b * b;
    }

    T df(T const *x, T *df_)
    {
        multimin_diff<T, Nrosenbrock<T>>(this, x, df_);
    }

    void fdf(T const *x, T *f_, T *df_)
    {
        *f_ = f(x);
        df(x, df_);
    }

    void initpt(T *x)
    {
        x[0] = -1.2;
        x[1] = 1.0;
    }
};

template <typename T>
class roth_fmin : public multimin_function<T, roth_fmin<T>>
{
  public:
    roth_fmin() { this->n = 2; }

    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = -13 + u + ((5 - v) * v - 2) * v;
        T b = -29 + u + ((v + 1) * v - 14) * v;
        return a * a + b * b;
    }

    void initpt(T *x)
    {
        x[0] = 4.5;
        x[1] = 3.5;
    }
};

template <typename T>
class roth : public multimin_function_fdf<T, roth<T>>
{
  public:
    roth() { this->n = 2; }

    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = -13 + u + ((5 - v) * v - 2) * v;
        T b = -29 + u + ((v + 1) * v - 14) * v;
        return a * a + b * b;
    }

    T df(T const *x, T *df_)
    {
        T u = x[0];
        T v = x[1];
        T a = -13 + u + ((5 - v) * v - 2) * v;
        T b = -29 + u + ((v + 1) * v - 14) * v;
        T c = -2 + v * (10 - 3 * v);
        T d = -14 + v * (2 + 3 * v);

        df_[0] = 2 * a + 2 * b;
        df_[1] = 2 * a * c + 2 * b * d;
    }

    void fdf(T const *x, T *f_, T *df_)
    {
        *f_ = f(x);
        df(x, df_);
    }

    void initpt(T *x)
    {
        x[0] = 4.5;
        x[1] = 3.5;
    }
};

template <typename T>
class Nroth : public multimin_function_fdf<T, Nroth<T>>
{
  public:
    Nroth() { this->n = 2; }

    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = -13 + u + ((5 - v) * v - 2) * v;
        T b = -29 + u + ((v + 1) * v - 14) * v;
        return a * a + b * b;
    }

    T df(T const *x, T *df_)
    {
        multimin_diff<T, Nroth<T>>(this, x, df_);
    }

    void fdf(T const *x, T *f_, T *df_)
    {
        *f_ = f(x);
        df(x, df_);
    }

    void initpt(T *x)
    {
        x[0] = 4.5;
        x[1] = 3.5;
    }
};

template <typename T>
class wood_fmin : public multimin_function<T, wood_fmin<T>>
{
  public:
    wood_fmin() { this->n = 4; }

    T f(T const *x)
    {
        T u1 = x[0];
        T u2 = x[1];
        T u3 = x[2];
        T u4 = x[3];

        T t1 = u1 * u1 - u2;
        T t2 = u3 * u3 - u4;

        return 100 * t1 * t1 + (1 - u1) * (1 - u1) + 90 * t2 * t2 + (1 - u3) * (1 - u3) + 10.1 * ((1 - u2) * (1 - u2) + (1 - u4) * (1 - u4)) + 19.8 * (1 - u2) * (1 - u4);
    }

    void initpt(T *x)
    {
        x[0] = -3;
        x[1] = -1;
        x[2] = 2;
        x[3] = 3;
    }
};

template <typename T>
class wood : public multimin_function_fdf<T, wood<T>>
{
  public:
    wood() { this->n = 4; }

    T f(T const *x)
    {
        T u1 = x[0];
        T u2 = x[1];
        T u3 = x[2];
        T u4 = x[3];

        T t1 = u1 * u1 - u2;
        T t2 = u3 * u3 - u4;

        return 100 * t1 * t1 + (1 - u1) * (1 - u1) + 90 * t2 * t2 + (1 - u3) * (1 - u3) + 10.1 * ((1 - u2) * (1 - u2) + (1 - u4) * (1 - u4)) + 19.8 * (1 - u2) * (1 - u4);
    }

    T df(T const *x, T *df_)
    {
        T u1 = x[0];
        T u2 = x[1];
        T u3 = x[2];
        T u4 = x[3];

        T t1 = u1 * u1 - u2;
        T t2 = u3 * u3 - u4;

        df_[0] = 400 * u1 * t1 - 2 * (1 - u1);
        df_[1] = -200 * t1 - 20.2 * (1 - u2) - 19.8 * (1 - u4);
        df_[2] = 360 * u3 * t2 - 2 * (1 - u3);
        df_[3] = -180 * t2 - 20.2 * (1 - u4) - 19.8 * (1 - u2);
    }

    void fdf(T const *x, T *f_, T *df_)
    {
        *f_ = f(x);
        df(x, df_);
    }

    void initpt(T *x)
    {
        x[0] = -3;
        x[1] = -1;
        x[2] = 2;
        x[3] = 3;
    }
};

template <typename T>
class Nwood : public multimin_function_fdf<T, Nwood<T>>
{
  public:
    Nwood() { this->n = 4; }

    T f(T const *x)
    {
        T u1 = x[0];
        T u2 = x[1];
        T u3 = x[2];
        T u4 = x[3];

        T t1 = u1 * u1 - u2;
        T t2 = u3 * u3 - u4;

        return 100 * t1 * t1 + (1 - u1) * (1 - u1) + 90 * t2 * t2 + (1 - u3) * (1 - u3) + 10.1 * ((1 - u2) * (1 - u2) + (1 - u4) * (1 - u4)) + 19.8 * (1 - u2) * (1 - u4);
    }

    T df(T const *x, T *df_)
    {
        multimin_diff<T, Nwood<T>>(this, x, df_);
    }

    void fdf(T const *x, T *f_, T *df_)
    {
        *f_ = f(x);
        df(x, df_);
    }

    void initpt(T *x)
    {
        x[0] = -3;
        x[1] = -1;
        x[2] = 2;
        x[3] = 3;
    }
};

template <typename T>
class spring_fmin : public multimin_function<T, spring_fmin<T>>
{
  public:
    spring_fmin() { this->n = 3; }

    T f(T const *x)
    {
        T x0 = x[0];
        T x1 = x[1];
        T x2 = x[2];

        T theta = std::atan2(x1, x0);
        T r = std::sqrt(x0 * x0 + x1 * x1);
        T z = x2;

        while (z > M_PI)
        {
            z -= M_2PI;
        }

        while (z < -M_PI)
        {
            z += M_2PI;
        }

        {
            T tmz = theta - z;
            T rm1 = r - 1;
            T ret = 0.1 * (std::expm1(tmz * tmz + rm1 * rm1) + std::abs(x2 / 10));
            return ret;
        }
    }

    void initpt(T *x)
    {
        x[0] = 1;
        x[1] = 0;
        x[2] = 7 * M_PI;
    }
};

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

typedef rosenbrock_fmin<double> Rosenbrock_f;
typedef rosenbrock<double> Rosenbrock;
typedef Nrosenbrock<double> NRosenbrock;
typedef roth_fmin<double> Roth_f;
typedef roth<double> Roth;
typedef Nroth<double> NRoth;
typedef wood_fmin<double> Wood_f;
typedef wood<double> Wood;
typedef Nwood<double> NWood;
typedef spring_fmin<double> Spring_f;

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

// TEST(multimin_test_vector_bfgs2, Handles_Rosenbrock_Function)
// {
//     typedef vector_bfgs2<double, Rosenbrock> TRosenbrock;
//     typedef vector_bfgs2<double, NRosenbrock> TNRosenbrock;

//     TRosenbrock t;
//     Rosenbrock r;

//     TNRosenbrock tn;
//     NRosenbrock rn;

//     EXPECT_TRUE((test_fdf<double, TRosenbrock, Rosenbrock>("Rosenbrock", &t, &r)));
//     EXPECT_TRUE((test_fdf<double, TNRosenbrock, NRosenbrock>("NRosenbrock", &tn, &rn)));
// }

// TEST(multimin_test_vector_bfgs2, Handles_Roth_Function)
// {
//     typedef vector_bfgs2<double, Roth> TRoth;
//     typedef vector_bfgs2<double, NRoth> TNRoth;

//     TRoth t;
//     Roth r;

//     TNRoth tn;
//     NRoth rn;

//     EXPECT_TRUE((test_fdf<double, TRoth, Roth>("Roth", &t, &r)));
//     EXPECT_TRUE((test_fdf<double, TNRoth, NRoth>("NRoth", &tn, &rn)));
// }

// TEST(multimin_test_vector_bfgs2, Handles_Wood_Function)
// {
//     typedef vector_bfgs2<double, Wood> TWood;
//     typedef vector_bfgs2<double, NWood> TNWood;

//     TWood t;
//     Wood r;

//     TNWood tn;
//     NWood rn;

//     EXPECT_TRUE((test_fdf<double, TWood, Wood>("Wood", &t, &r)));
//     EXPECT_TRUE((test_fdf<double, TNWood, NWood>("NWood", &tn, &rn)));
// }



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
