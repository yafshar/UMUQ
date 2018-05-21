#ifndef UMUQ_OPTIMIZATIONTESTFUNCTIONS_H
#define UMUQ_OPTIMIZATIONTESTFUNCTIONS_H

namespace multimin
{

/*! \class rosenbrock_fmin
 * \brief The Rosenbrock function
 * 
 * The Rosenbrock function, also referred to as the Valley or Banana function, is a 
 * popular test problem for gradient-based optimization algorithms. 
 * The function is unimodal, and the global minimum lies in a narrow, parabolic valley. 
 * However, even though this valley is easy to find, convergence to the minimum is 
 * difficult (Picheny et al., 2012).
 */
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

/*! \class rosenbrock
 * \brief The Rosenbrock function
 * 
 * The Rosenbrock function, also referred to as the Valley or Banana function, is a 
 * popular test problem for gradient-based optimization algorithms. 
 * The function is unimodal, and the global minimum lies in a narrow, parabolic valley. 
 * However, even though this valley is easy to find, convergence to the minimum is 
 * difficult (Picheny et al., 2012).
 */
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
        return 0;
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

/*! \class Nrosenbrock
 * \brief The Rosenbrock function
 * 
 * The Rosenbrock function, also referred to as the Valley or Banana function, is a 
 * popular test problem for gradient-based optimization algorithms. 
 * The function is unimodal, and the global minimum lies in a narrow, parabolic valley. 
 * However, even though this valley is easy to find, convergence to the minimum is 
 * difficult (Picheny et al., 2012).
 */
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
        return multimin_diff<T, Nrosenbrock<T>>(this, x, df_);
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
        return 0;
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
        return multimin_diff<T, Nroth<T>>(this, x, df_);
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
        return 0;
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
        return multimin_diff<T, Nwood<T>>(this, x, df_);
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

} //namespace multimin

#endif
