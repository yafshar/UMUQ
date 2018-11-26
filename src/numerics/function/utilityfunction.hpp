#ifndef UMUQ_UTILITYFUNCTION_H
#define UMUQ_UTILITYFUNCTION_H

#include "linearfunctionwrapper.hpp"

namespace umuq
{

inline namespace multimin
{

/*!
 * \ingroup Multimin_Module
 * 
 * \brief Helper routines
 *
 */

/*!
 * \ingroup Multimin_Module
 * 
 * \brief Finds the real roots of \f$ a x^2 + b x + c = 0 \f$
 * 
 * \tparam DataType 
 * \param a 
 * \param b 
 * \param c 
 * \param x0 
 * \param x1 
 * 
 * \returns int Number of found roots
 */
template <typename DataType>
int poly_solve_quadratic(DataType const a, DataType const b, DataType const c, DataType *x0, DataType *x1)
{
    // Handle linear case
    if (a == DataType{})
    {
        if (b == DataType{})
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
        DataType disc = b * b - 4 * a * c;

        if (disc > DataType{})
        {
            if (b == DataType{})
            {
                DataType r = std::sqrt(-c / a);

                *x0 = -r;
                *x1 = r;
            }
            else
            {
                DataType sgnb = (b > DataType{} ? 1 : -1);
                DataType temp = -0.5 * (b + sgnb * std::sqrt(disc));
                DataType r1 = temp / a;
                DataType r2 = c / temp;

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
        else if (disc == DataType{})
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
 * \ingroup Multimin_Module
 * 
 * 
 */

/*!
 * \ingroup Multimin_Module
 * 
 * \brief Find a minimum in \f$ x=[0,1] \f$ of the interpolating quadratic through
 * \f$(0,f0)\f$, and \f$ (1,f1) \f$ with derivative \f$ fp0 \f$ at \f$x = 0. \f$  
 * The interpolating polynomial is \f$ q(x) = f0 + fp0 * z + (f1-f0-fp0) * z^2 \f$
 * 
 * \tparam DataType Data type
 * 
 * \param f0 
 * \param fp0 
 * \param f1 
 * \param zl 
 * \param zh 
 * 
 * \returns DataType Minimum found value in \f$ x=[0,1] \f$ of the interpolating quadratic 
 */
template <typename DataType>
DataType interp_quad(DataType const f0, DataType const fp0, DataType const f1, DataType const zl, DataType const zh)
{
    DataType fl = f0 + zl * (fp0 + zl * (f1 - f0 - fp0));
    DataType fh = f0 + zh * (fp0 + zh * (f1 - f0 - fp0));

    // Curvature
    DataType c = 2 * (f1 - f0 - fp0);

    DataType zmin = zl;
    DataType fmin = fl;

    if (fh < fmin)
    {
        zmin = zh;
        fmin = fh;
    }

    // Positive curvature required for a minimum
    if (c > DataType{})
    {
        // Location of minimum
        DataType z = -fp0 / c;

        if (z > zl && z < zh)
        {
            DataType f = f0 + z * (fp0 + z * (f1 - f0 - fp0));
            if (f < fmin)
            {
                zmin = z;
                fmin = f;
            };
        }
    }

    return zmin;
}

/*!
 * \ingroup Multimin_Module
 * 
 * \brief Find a minimum in \f$ x=[0,1] \f$ of the interpolating cubic through
 * \f$ (0,f0) \f$, and \f$ (1,f1) \f$ with derivatives \f$ fp0 \f$ at \f$ x=0 \f$ 
 * and \f$ fp1 \f$ at \f$ x=1. \f$
 * 
 * \tparam DataType Data type
 * 
 * \param c0 
 * \param c1 
 * \param c2 
 * \param c3 
 * \param z 
 * 
 * \returns DataType Minimum found value in \f$ x=[0,1] \f$ of the interpolating cubic
 * 
 *
 * Find a minimum in \f$ x=[0,1] \f$ of the interpolating cubic through
 * \f$ (0,f0) \f$, and \f$ (1,f1) \f$ with derivatives \f$ fp0 \f$ at \f$ x=0 \f$ 
 * and \f$ fp1 \f$ at \f$ x=1. \f$
 *
 * The interpolating polynomial is:<br>
 * 
 * \f$ c(x) = f0 + fp0 * z + \eta * z^2 + xi * z^3 \f$, where 
 * \f$
 * \begin{aligned}
 * \nonumber \eta=3*(f1-f0)-2*fp0-fp1, \\
 * \nonumber xi=fp0+fp1-2*(f1-f0). 
 * \end{aligned}
 * \f$
 */
template <typename DataType>
inline DataType cubic(DataType const c0, DataType const c1, DataType const c2, DataType const c3, DataType const z)
{
    return c0 + z * (c1 + z * (c2 + z * c3));
}

/*!
 * \ingroup Multimin_Module
 * 
 * \brief Check for the extremum
 * 
 * \tparam DataType Data type
 *  
 * \param c0 
 * \param c1 
 * \param c2 
 * \param c3 
 * \param z 
 * \param zmin 
 * \param fmin 
 */
template <typename DataType>
inline void check_extremum(DataType const c0, DataType const c1, DataType const c2, DataType const c3, DataType const z, DataType *zmin, DataType *fmin)
{
    // Could make an early return by testing curvature >0 for minimum
    DataType y = cubic<DataType>(c0, c1, c2, c3, z);

    if (y < *fmin)
    {
        // Accepted new point
        *zmin = z;
        *fmin = y;
    }
}

/*!
 * \ingroup Multimin_Module
 * 
 * \brief A cubic polynomial interpolation using \f$ f(\alpha_i), \acute{f}(\alpha_i), f(\alpha_{i-1}), and \acute{f}(\alpha_{i-1}) \f$.
 * 
 * \tparam DataType Data type
 * 
 * \param f0   \f$ f(\alpha_i) \f$
 * \param fp0  \f$ \acute{f}(\alpha_i) \f$
 * \param f1   \f$ f(\alpha_{i-1}) \f$
 * \param fp1  \f$ \acute{f}(\alpha_{i-1}) \f$
 * \param zl   Lower bound 
 * \param zh   Higher bound
 * 
 * \return DataType Interpolation value
 */
template <typename DataType>
DataType interp_cubic(DataType const f0, DataType const fp0, DataType const f1, DataType const fp1, DataType const zl, DataType const zh)
{
    DataType eta = 3 * (f1 - f0) - 2 * fp0 - fp1;
    DataType xi = fp0 + fp1 - 2 * (f1 - f0);
    DataType c0 = f0;
    DataType c1 = fp0;
    DataType c2 = eta;
    DataType c3 = xi;
    DataType z0;
    DataType z1;

    DataType zmin = zl;

    DataType fmin = cubic<DataType>(c0, c1, c2, c3, zl);

    check_extremum<DataType>(c0, c1, c2, c3, zh, &zmin, &fmin);

    switch (poly_solve_quadratic<DataType>(3 * c3, 2 * c2, c1, &z0, &z1))
    {
    // Found 2 roots
    case (2):
        if (z0 > zl && z0 < zh)
        {
            check_extremum<DataType>(c0, c1, c2, c3, z0, &zmin, &fmin);
        }

        if (z1 > zl && z1 < zh)
        {
            check_extremum<DataType>(c0, c1, c2, c3, z1, &zmin, &fmin);
        }
        break;
    // Found 1 root
    case (1):
        if (z0 > zl && z0 < zh)
        {
            check_extremum<DataType>(c0, c1, c2, c3, z0, &zmin, &fmin);
        }
        break;
    }

    return zmin;
}

/*!
 * \ingroup Multimin_Module
 * 
 * \brief interpolate
 * 
 * \tparam DataType Data type
 * 
 * \param a 
 * \param fa 
 * \param fpa 
 * \param b 
 * \param fb 
 * \param fpb 
 * \param xmin 
 * \param xmax 
 * \param order 
 * 
 * \returns DataType Interpolation value
 */
template <typename DataType>
DataType interpolate(DataType const a, DataType const fa, DataType const fpa, DataType const b, DataType const fb, DataType const fpb, DataType const xmin, DataType const xmax, int const order)
{
    // Map [a,b] to [0,1]
    DataType zmin = (xmin - a) / (b - a);
    DataType zmax = (xmax - a) / (b - a);

    if (zmin > zmax)
    {
        std::swap(zmin, zmax);
    }

    DataType z;
    if (order > 2 && std::isfinite(fpb))
    {
        z = interp_cubic<DataType>(fa, fpa * (b - a), fb, fpb * (b - a), zmin, zmax);
    }
    else
    {
        z = interp_quad<DataType>(fa, fpa * (b - a), fb, zmin, zmax);
    }

    return a + z * (b - a);
}

/*!
 * \ingroup Multimin_Module
 * 
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
 * \tparam DataType Data type
 * 
 * \param obj     linearFunctionWrapper object
 * \param rho     Fixed parameter defined by \f$ f(\alpha) \ge f(0) + \alpha (1 - \rho) \acute{f}(0) \f$  
 * \param sigma   Fixed parameter defined by \f$ \acute{f}(\alpha) \ge \sigma \acute{f}(0), ~~ \sigma \in (\rho,1) \f$
 * \param tau1    Preset factor by which the size of the jumps is increased
 * \param tau2    Preset factor
 * \param tau3    Preset factor
 * \param order   
 * \param alpha1  \f$ \alpha_1 = \min{(1, -2\Delta f/\acute{f}(0))} \f$
 * \param alpha_new 
 * 
 * \return true 
 * \return false 
 */
template <typename DataType>
bool minimize(linearFunctionWrapper<DataType> &obj,
              DataType const rho, DataType const sigma,
              DataType const tau1, DataType const tau2,
              DataType const tau3, int const order,
              DataType const alpha1, DataType *alpha_new)
{
    DataType f0;
    DataType fp0;
    DataType falpha;
    DataType falpha_prev;
    DataType fpalpha;
    DataType fpalpha_prev;
    DataType delta;
    DataType alpha_next;
    DataType alpha(alpha1);
    DataType alpha_prev(0);
    DataType a(0);
    DataType b(alpha);
    DataType fa;
    DataType fb(0);
    DataType fpa;
    DataType fpb(0);

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
            fpb = std::numeric_limits<DataType>::quiet_NaN();

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

        if (fpalpha >= DataType{})
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
            DataType lower = alpha + delta;
            DataType upper = alpha + tau1 * delta;

            alpha_next = interpolate<DataType>(alpha_prev, falpha_prev, fpalpha_prev, alpha, falpha, fpalpha, lower, upper, order);
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
            DataType lower = a + tau2 * delta;
            DataType upper = b - tau3 * delta;

            alpha = interpolate<DataType>(a, fa, fpa, b, fb, fpb, lower, upper, order);
        }

        falpha = obj.f(alpha);

        if ((a - alpha) * fpa <= std::numeric_limits<DataType>::epsilon())
        {
            // Roundoff prevents progress
            UMUQFAILRETURN("The minimizer is unable to improve on its current estimate, either due \n to the numerical difficulty or because a genuine local minimum has been reached!");
        }

        if (falpha > f0 + rho * alpha * fp0 || falpha >= fa)
        {
            // \f$ a_next = a \f$
            b = alpha;
            fb = falpha;
            fpb = std::numeric_limits<DataType>::quiet_NaN();
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

            if (((b - a) >= DataType{} && fpalpha >= DataType{}) || ((b - a) <= DataType{} && fpalpha <= DataType{}))
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

} // namespace multimin
} // namespace umuq

#endif // UMUQ_UTILITYFUNCTION
