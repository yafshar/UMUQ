#ifndef UMUQ_OPTIMIZATIONTESTFUNCTIONS_H
#define UMUQ_OPTIMIZATIONTESTFUNCTIONS_H

#include "numerics/multimin.hpp"
/*!
 * \brief The Rosenbrock function
 * 
 * The Rosenbrock function, also referred to as the Valley or Banana function, is a 
 * popular test problem for gradient-based optimization algorithms. 
 * The function is unimodal, and the global minimum lies in a narrow, parabolic valley. 
 * However, even though this valley is easy to find, convergence to the minimum is 
 * difficult (Picheny et al., 2012).
 * 
 * 
 * \f$ f(x,y) = 100(y - x^2)^2 + (1 - x)^2 \f$
 * Minimum: 0 at (1,1)
 */

/*!
 * \brief Rosenbrock function
 * 
 * \param x  2-D Input point
 *  
 * \return Function value at x 
 */
double rosenbrock_f(double const *x)
{
    double u = x[0];
    double v = x[1];
    double a = u - 1;
    double b = u * u - v;

    return a * a + 10 * b * b;
}

/*!
 * \brief Derivative of Rosenbrock function
 * 
 * \param x   2-D Input point
 * \param df  Function derivative \f$ \frac{\partial f}{\partial x_0} \text{and} \frac{\partial f}{\partial x_1} \f$
 * 
 * \return true 
 * \return false 
 */
bool rosenbrock_df(double const *x, double *df)
{
    double u = x[0];
    double v = x[1];
    double b = u * u - v;

    df[0] = 2 * (u - 1) + 40 * u * b;
    df[1] = -20 * b;

    return true;
}

/*!
 * \brief Rosenbrock function & it's derivative
 * 
 * \param x   2-D Input point
 * \param f   Function value at point x
 * \param df  Function derivative \f$ \frac{\partial f}{\partial x_0} \text{and} \frac{\partial f}{\partial x_1} \f$
 * 
 * \return true 
 * \return false 
 */
bool rosenbrock_fdf(double const *x, double *f, double *df)
{
    double u = x[0];
    double v = x[1];
    double a = u - 1;
    double b = u * u - v;

    *f = a * a + 10 * b * b;

    df[0] = 2 * (u - 1) + 40 * u * b;
    df[1] = -20 * b;

    return true;
}

/*!
 * \brief The Freudenstein-Roth's Function
 * 
 * The Freudenstein-Roth's Function, also referred to roth function.
 * 
 * \f$ f(x,y) = (-13 + x + ((5 - y)y - 2)y)^2 + (-29 + x + ((y + 1)y - 14)y)^2 \f$
 * Minimum: 0 at (5,4) and 48.9842 at (11.41, -0.8986)
 */

/*!
 * \brief Roth function
 * 
 * \param x  2-D Input point
 *  
 * \return Function value at x 
 */
double roth_f(double const *x)
{
    double u = x[0];
    double v = x[1];

    double a = -13 + u + ((5 - v) * v - 2) * v;
    double b = -29 + u + ((v + 1) * v - 14) * v;

    return a * a + b * b;
}

/*!
 * \brief Derivative of Roth function
 * 
 * \param x   2-D Input point
 * \param df  Function derivative \f$ \frac{\partial f}{\partial x_0} \text{and} \frac{\partial f}{\partial x_1} \f$
 * 
 * \return true 
 * \return false 
 */
bool roth_df(double const *x, double *df)
{
    double u = x[0];
    double v = x[1];
    double a = -13 + u + ((5 - v) * v - 2) * v;
    double b = -29 + u + ((v + 1) * v - 14) * v;
    double c = -2 + v * (10 - 3 * v);
    double d = -14 + v * (2 + 3 * v);

    df[0] = 2 * a + 2 * b;
    df[1] = 2 * a * c + 2 * b * d;

    return true;
}

/*!
 * \brief Roth function & it's derivative
 * 
 * \param x   2-D Input point
 * \param f   Function value at point x
 * \param df  Function derivative \f$ \frac{\partial f}{\partial x_0} \text{and} \frac{\partial f}{\partial x_1} \f$
 * 
 * \return true 
 * \return false 
 */
bool roth_fdf(double const *x, double *f, double *df)
{
    double u = x[0];
    double v = x[1];
    double a = -13 + u + ((5 - v) * v - 2) * v;
    double b = -29 + u + ((v + 1) * v - 14) * v;
    double c = -2 + v * (10 - 3 * v);
    double d = -14 + v * (2 + 3 * v);

    *f = a * a + b * b;

    df[0] = 2 * a + 2 * b;
    df[1] = 2 * a * c + 2 * b * d;

    return true;
}

/*!
 * \brief The Wood's (or Colville's) function
 * 
 * The Wood's (or Colville's) function, also referred to wood function.
 * 
 * 
 * \f$ f = 100*(x_1^2-x_2)^2+(x_1-1)^2+(x_3-1)^2+90*(x_3^2-x_4)^2+10.1*((x_2-1)^2+(x_4-1)^2)+19.8*(x_2^-1)*(x_4-1) \f$
 * Minimum: 0 at (1,1,1,1)
 *  
 * The function is usually evaluated on the hypercube \f$ x \in [-10, 10]~\text{for all}~i = 1, 2, 3, 4. \f$
 */

/*!
 * \brief Wood function
 * 
 * \param x  4-D Input point
 *  
 * \return Function value at x 
 */
double wood_f(double const *x)
{
    double u1 = x[0];
    double u2 = x[1];
    double u3 = x[2];
    double u4 = x[3];

    double t1 = u1 * u1 - u2;
    double t2 = u3 * u3 - u4;

    return 100 * t1 * t1 + (1 - u1) * (1 - u1) + 90 * t2 * t2 + (1 - u3) * (1 - u3) + 10.1 * ((1 - u2) * (1 - u2) + (1 - u4) * (1 - u4)) + 19.8 * (1 - u2) * (1 - u4);
}

/*!
 * \brief Derivative of Wood function
 * 
 * \param x   4-D Input point
 * \param df  Function derivative \f$ \frac{\partial f}{\partial x_0}, \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \text{and} \frac{\partial f}{\partial x_3} \f$
 * 
 * \return true 
 * \return false 
 */
bool wood_df(double const *x, double *df)
{
    double u1 = x[0];
    double u2 = x[1];
    double u3 = x[2];
    double u4 = x[3];

    double t1 = u1 * u1 - u2;
    double t2 = u3 * u3 - u4;

    df[0] = 400 * u1 * t1 - 2 * (1 - u1);
    df[1] = -200 * t1 - 20.2 * (1 - u2) - 19.8 * (1 - u4);
    df[2] = 360 * u3 * t2 - 2 * (1 - u3);
    df[3] = -180 * t2 - 20.2 * (1 - u4) - 19.8 * (1 - u2);

    return true;
}

/*!
 * \brief Wood function & it's derivative
 * 
 * \param x    4-D Input point
 * \param f    Function value at point x
 * \param df   Function derivative \f$ \frac{\partial f}{\partial x_0}, \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \text{and} \frac{\partial f}{\partial x_3} \f$
 * 
 * \return true 
 * \return false 
 */
bool wood_fdf(double const *x, double *f, double *df)
{
    double u1 = x[0];
    double u2 = x[1];
    double u3 = x[2];
    double u4 = x[3];

    double t1 = u1 * u1 - u2;
    double t2 = u3 * u3 - u4;

    *f = 100 * t1 * t1 + (1 - u1) * (1 - u1) + 90 * t2 * t2 + (1 - u3) * (1 - u3) + 10.1 * ((1 - u2) * (1 - u2) + (1 - u4) * (1 - u4)) + 19.8 * (1 - u2) * (1 - u4);

    df[0] = 400 * u1 * t1 - 2 * (1 - u1);
    df[1] = -200 * t1 - 20.2 * (1 - u2) - 19.8 * (1 - u4);
    df[2] = 360 * u3 * t2 - 2 * (1 - u3);
    df[3] = -180 * t2 - 20.2 * (1 - u4) - 19.8 * (1 - u2);

    return true;
}

/*!
 * \brief The Spring function
 * 
 */

/*!
 * \brief Spring function
 * 
 * \param x  3-D Input point
 *  
 * \return Function value at x 
 */
double spring_f(double const *x)
{
    double x0 = x[0];
    double x1 = x[1];
    double x2 = x[2];

    double theta = std::atan2(x1, x0);
    double r = std::sqrt(x0 * x0 + x1 * x1);
    double z = x2;

    while (z > M_PI)
    {
        z -= M_2PI;
    }

    while (z < -M_PI)
    {
        z += M_2PI;
    }

    double tmz = theta - z;
    double rm1 = r - 1;

    return 0.1 * (std::expm1(tmz * tmz + rm1 * rm1) + std::abs(x2 / 10));
}

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
    /*!
     * \brief Construct a new rosenbrock fmin object
     * 
     */
    rosenbrock_fmin() { this->n = 2; }

    /*!
     * \brief function definition
     * 
     * \param  x input
     * \return f
     */
    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = u - 1;
        T b = u * u - v;
        return a * a + 10 * b * b;
    }

    /*!
     * \brief Set the initial point
     * 
     * \param x input data point
     */
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
    /*!
     * \brief Construct a new rosenbrock object
     * 
     */
    rosenbrock() { this->n = 2; }

    /*!
     * \brief function definition
     * 
     * \param  x input data 
     * \return f
     */
    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = u - 1;
        T b = u * u - v;
        return a * a + 10 * b * b;
    }

    /*!
     * \brief derivative of a function
     * 
     * \param x   input data 
     * \param df_ function derivative \f$ \frac{\partial f}{\partial x_0} \text{and} \frac{\partial f}{\partial x_1} \f$
     */
    T df(T const *x, T *df_)
    {
        T u = x[0];
        T v = x[1];
        T b = u * u - v;

        df_[0] = 2 * (u - 1) + 40 * u * b;
        df_[1] = -20 * b;
        return 0;
    }

    /*!
     * \brief function and its derivative
     * 
     * \param x   input data
     * \param f_  function
     * \param df_ derivative
     */
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

    /*!
     * \brief Set the default initial point
     * 
     * \param x input data point
     */
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

    /*!
     * \brief function
     * 
     * \param x input data
     * \return  the function value at point x
     */
    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = u - 1;
        T b = u * u - v;
        return a * a + 10 * b * b;
    }

    /*!
     * \brief derivative
     *  
     * \param x   input data
     * \param df_ derivative at point x
     * \return T  derivative at point x
     */
    T df(T const *x, T *df_)
    {
        return multimin_diff<T, Nrosenbrock<T>>(this, x, df_);
    }

    /*!
     * \brief function and its derivative
     * 
     * \param x   input point
     * \param f_  function at point x
     * \param df_ derivative at point x
     */
    void fdf(T const *x, T *f_, T *df_)
    {
        *f_ = f(x);
        df(x, df_);
    }

    /*!
     * \brief Set the default initial point
     * 
     * \param x input data point
     */
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
    /*!
     * \brief Construct a new roth fmin object
     * 
     */
    roth_fmin() { this->n = 2; }

    /*!
     * \brief function
     * 
     * \param x input data
     * \return  the function value at point x
     */
    T f(T const *x)
    {
        T u = x[0];
        T v = x[1];
        T a = -13 + u + ((5 - v) * v - 2) * v;
        T b = -29 + u + ((v + 1) * v - 14) * v;
        return a * a + b * b;
    }

    /*!
     * \brief Set the default initial point
     * 
     * \param x input data point
     */
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
