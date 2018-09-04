#ifndef UMUQ_UTILITYFUNCTION_H
#define UMUQ_UTILITYFUNCTION_H

#include "linearfunctionwrapper.hpp"

/*!
 * \brief Helper routines
 * \ingroup multimin_Module
 *
 */

/*!
 * \brief finds the real roots of \f$ a x^2 + b x + c = 0 \f$
 * 
 * \returns number of found roots
 */
template <typename T>
int poly_solve_quadratic(T const a, T const b, T const c, T *x0, T *x1)
{
    // Handle linear case
    if (a == T{})
    {
        if (b == T{})
        {
            // number of found roots is 0
            return 0;
        }
        else
        {
            *x0 = -c / b;

            // number of found roots is 1
            return 1;
        };
    }

    {
        T disc = b * b - 4 * a * c;

        if (disc > T{})
        {
            if (b == T{})
            {
                T r = std::sqrt(-c / a);

                *x0 = -r;
                *x1 = r;
            }
            else
            {
                T sgnb = (b > T{} ? 1 : -1);
                T temp = -0.5 * (b + sgnb * std::sqrt(disc));
                T r1 = temp / a;
                T r2 = c / temp;

                if (r1 < r2)
                {
                    *x0 = r1;
                    *x1 = r2;
                }
                else
                {
                    *x0 = r2;
                    *x1 = r1;
                }
            }

            // number of found roots is 2
            return 2;
        }
        else if (disc == T{})
        {
            *x0 = -0.5 * b / a;
            *x1 = -0.5 * b / a;

            // number of found roots is 2
            return 2;
        }
        else
        {
            // number of found roots is 0
            return 0;
        }
    }
}

/*! 
 * Find a minimum in \f$ x=[0,1] \f$ of the interpolating quadratic through
 * (0,f0) (1,f1) with derivative fp0 at x=0.  The interpolating
 * polynomial is \f$ q(x) = f0 + fp0 * z + (f1-f0-fp0) * z^2 \f$
 */
template <typename T>
T interp_quad(T const f0, T const fp0, T const f1, T const zl, T const zh)
{
    T fl = f0 + zl * (fp0 + zl * (f1 - f0 - fp0));
    T fh = f0 + zh * (fp0 + zh * (f1 - f0 - fp0));

    // Curvature
    T c = 2 * (f1 - f0 - fp0);

    T zmin = zl;
    T fmin = fl;

    if (fh < fmin)
    {
        zmin = zh;
        fmin = fh;
    }

    // Positive curvature required for a minimum
    if (c > T{})
    {
        // Location of minimum
        T z = -fp0 / c;

        if (z > zl && z < zh)
        {
            T f = f0 + z * (fp0 + z * (f1 - f0 - fp0));
            if (f < fmin)
            {
                zmin = z;
                fmin = f;
            };
        }
    }

    return zmin;
}

/* 
 * Find a minimum in x=[0,1] of the interpolating cubic through
 * (0,f0) (1,f1) with derivatives fp0 at x=0 and fp1 at x=1.
 *
 * The interpolating polynomial is:
 *
 * \f$ c(x) = f0 + fp0 * z + eta * z^2 + xi * z^3 \f$
 *
 * where \f$ eta=3*(f1-f0)-2*fp0-fp1, \f$
 *       \f$ xi=fp0+fp1-2*(f1-f0). \f$
 */
template <typename T>
inline T cubic(T const c0, T const c1, T const c2, T const c3, T const z)
{
    return c0 + z * (c1 + z * (c2 + z * c3));
}

template <typename T>
inline void check_extremum(T const c0, T const c1, T const c2, T const c3, T const z, T *zmin, T *fmin)
{
    // Could make an early return by testing curvature >0 for minimum
    T y = cubic<T>(c0, c1, c2, c3, z);

    if (y < *fmin)
    {
        // Accepted new point
        *zmin = z;
        *fmin = y;
    }
}

/*!
 * \brief A cubic polynomial interpolation using \f$ f(\alpha_i), \acute{f}(\alpha_i), f(\alpha_{i-1}), and \acute{f}(\alpha_{i-1}) \f$.
 * 
 * \tparam T Data type
 * 
 * \param f0   \f$ f(\alpha_i) \f$
 * \param fp0  \f$ \acute{f}(\alpha_i) \f$
 * \param f1   \f$ f(\alpha_{i-1}) \f$
 * \param fp1  \f$ \acute{f}(\alpha_{i-1}) \f$
 * \param zl   Lower bound 
 * \param zh   Higher bound
 * 
 * \return Interpolation value
 */
template <typename T>
T interp_cubic(T const f0, T const fp0, T const f1, T const fp1, T const zl, T const zh)
{
    T eta = 3 * (f1 - f0) - 2 * fp0 - fp1;
    T xi = fp0 + fp1 - 2 * (f1 - f0);
    T c0 = f0;
    T c1 = fp0;
    T c2 = eta;
    T c3 = xi;
    T z0;
    T z1;

    T zmin = zl;

    T fmin = cubic<T>(c0, c1, c2, c3, zl);
    
    check_extremum<T>(c0, c1, c2, c3, zh, &zmin, &fmin);

    switch (poly_solve_quadratic<T>(3 * c3, 2 * c2, c1, &z0, &z1))
    {
    // Found 2 roots
    case (2):
        if (z0 > zl && z0 < zh)
        {
            check_extremum<T>(c0, c1, c2, c3, z0, &zmin, &fmin);
        }

        if (z1 > zl && z1 < zh)
        {
            check_extremum<T>(c0, c1, c2, c3, z1, &zmin, &fmin);
        }
        break;
    // Found 1 root
    case (1):
        if (z0 > zl && z0 < zh)
        {
            check_extremum<T>(c0, c1, c2, c3, z0, &zmin, &fmin);
        }
        break;
    }

    return zmin;
}

/*!
 *  
 *  
 */
template <typename T>
T interpolate(T const a, T const fa, T const fpa, T const b, T const fb, T const fpb, T const xmin, T const xmax, int const order)
{
    // Map [a,b] to [0,1]
    T zmin = (xmin - a) / (b - a);
    T zmax = (xmax - a) / (b - a);

    if (zmin > zmax)
    {
        std::swap(zmin, zmax);
    }

    T z;
    if (order > 2 && std::isfinite(fpb))
    {
        z = interp_cubic<T>(fa, fpa * (b - a), fb, fpb * (b - a), zmin, zmax);
    }
    else
    {
        z = interp_quad<T>(fa, fpa * (b - a), fb, zmin, zmax);
    }

    return a + z * (b - a);
}

/*!
 * \brief Line search algorithm for general unconstrained minimization
 * 
 * An assumption for this algorithm is that the line search can be restricted to the 
 * interval \f$ (0, \mu] \f$ where \f$ \mu = \frac{\bar{f}-f(0)}{\rho \acute{f}(0)} \f$
 * is the point at which the \f$\rho-\text{line}\f$ intersects the line \f$ f = \bar{f} \f$.
 * 
 * After the bracketing phase has achieved its aim of bracketing an interval of acceptable 
 * points, sectioning phase generates a sequence of brackets whose lengths tend to zero.
 * 
 * In this algorithm \f$ \tau_1 > 1 \f$ is a preset factor by which the size of the jumps is
 * increased, typically \f$ \tau_1 = 9 \f$. 
 * \f$ \tau_2 \f$ and \f$ \tau_3 \f$ are preset factors \f$  0 < \tau_2  < \tau_3 \le \frac{1}{2} \f$.
 * 
 * Typical values are \f$ \tau_2 = \frac{1}{10} \f$ ( \f$ (\tau_2 \le \sigma \text{is advisable}) \f$  
 * and \f$ \tau_3 = \frac{1}{2} \f$, although the algorithm is insensitive to the precise values that
 * are used. 
 * 
 * Recommended values from Fletcher: 
 * \f$ \rho = 0.01, \sigma = 0.1, \tau_1 = 9, \tau_2 = 0.05, \tau_3 = 0.5 \f$ 
 * 
 * Ref:
 * R. Fletcher, Practical Methods of Optimization (Second Edition) Wiley (1987), ISBN 0471915475.
 * 
 * \tparam T      Data type
 * 
 * \param obj     linearFunctionWrapper object
 * \param rho     Fixed parameter defined by \f$ f(\alpha) \ge f(0) + \alpha (1 - \rho) \acute{f}(0) \f$  
 * \param sigma   Fixed parameter defined by \f$ \acute{f}(\alpha) \ge \sigma \acute{f}(0), ~~ \sigma \in (\rho,1) \f$
 * \param tau1    Preset factor by which the size of the jumps is increased
 * \param tau2    Preset factor
 * \param tau3    Preset factor
 * \param order   
 * \param alpha1  \f$ \alpha_1 = \min{(1, -2\Deltaf/\acute{f}(0))} \f$
 * \param alpha_new 
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool minimize(linearFunctionWrapper<T> &obj,
              T const rho, T const sigma,
              T const tau1, T const tau2,
              T const tau3, int const order,
              T const alpha1, T *alpha_new)
{
    T f0;
    T fp0;
    T falpha;
    T falpha_prev;
    T fpalpha;
    T fpalpha_prev;
    T delta;
    T alpha_next;
    T alpha(alpha1);
    T alpha_prev(0);
    T a(0);
    T b(alpha);
    T fa;
    T fb(0);
    T fpa;
    T fpb(0);

    if (!obj.fdf(0, &f0, &fp0))
    {
        UMUQFAILRETURN("Failed to compute the function!");
    }

    falpha_prev = f0;
    fpalpha_prev = fp0;

    // Avoid uninitialized variables morning
    b = alpha;
    fa = f0;
    fpa = fp0;

    int i = 0;

    // Begin bracketing
    while (i++ < 100)
    {
        // TO CHECK
        falpha = obj.f(alpha);

        // Fletcher's rho test

        if (falpha > f0 + alpha * rho * fp0 || falpha >= falpha_prev)
        {
            a = alpha_prev;
            fa = falpha_prev;
            fpa = fpalpha_prev;
            b = alpha;
            fb = falpha;
            fpb = std::numeric_limits<T>::quiet_NaN();

            // Goto sectioning
            break;
        }

        if (!obj.df(alpha, &fpalpha))
        {
            UMUQFAILRETURN("Failed to compute the function!");
        }

        // Fletcher's sigma test
        if (std::abs(fpalpha) <= -sigma * fp0)
        {
            *alpha_new = alpha;
            return true;
        }

        if (fpalpha >= T{})
        {
            a = alpha;
            fa = falpha;
            fpa = fpalpha;
            b = alpha_prev;
            fb = falpha_prev;
            fpb = fpalpha_prev;

            // Goto sectioning
            break;
        }

        delta = alpha - alpha_prev;

        {
            T lower = alpha + delta;
            T upper = alpha + tau1 * delta;

            alpha_next = interpolate<T>(alpha_prev, falpha_prev, fpalpha_prev, alpha, falpha, fpalpha, lower, upper, order);
        }

        alpha_prev = alpha;
        falpha_prev = falpha;
        fpalpha_prev = fpalpha;
        alpha = alpha_next;
    }

    // Sectioning of bracket \f$ [a,b] \f$
    while (i++ < 100)
    {
        delta = b - a;

        {
            T lower = a + tau2 * delta;
            T upper = b - tau3 * delta;

            alpha = interpolate<T>(a, fa, fpa, b, fb, fpb, lower, upper, order);
        }

        falpha = obj.f(alpha);

        if ((a - alpha) * fpa <= std::numeric_limits<T>::epsilon())
        {
            // Roundoff prevents progress
            UMUQFAILRETURN("The minimizer is unable to improve on its current estimate, either due \n to the numerical difficulty or because a genuine local minimum has been reached!");
        }

        if (falpha > f0 + rho * alpha * fp0 || falpha >= fa)
        {
            // \f$ a_next = a \f$
            b = alpha;
            fb = falpha;
            fpb = std::numeric_limits<T>::quiet_NaN();
        }
        else
        {
            if (!obj.df(alpha, &fpalpha))
            {
                UMUQFAILRETURN("Failed to compute the function!")
            }

            if (std::abs(fpalpha) <= -sigma * fp0)
            {
                *alpha_new = alpha;

                // Terminate
                return true;
            }

            if (((b - a) >= T{} && fpalpha >= T{}) || ((b - a) <= T{} && fpalpha <= T{}))
            {
                b = a;
                fb = fa;
                fpb = fpa;
                a = alpha;
                fa = falpha;
                fpa = fpalpha;
            }
            else
            {
                a = alpha;
                fa = falpha;
                fpa = fpalpha;
            }
        }
    }

    return true;
}

#endif // UMUQ_UTILITYFUNCTION
