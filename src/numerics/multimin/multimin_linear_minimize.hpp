#ifndef UMUQ_MULTIMIN_LINEAR_MINIMIZE_H
#define UMUQ_MULTIMIN_LINEAR_MINIMIZE_H

/*!
 * \brief Helper routines
 * \ingroup multimin_Module
 *
 */

//TODO : Documentation is missing!

/*!
 * \brief finds the real roots of \f$ a x^2 + b x + c = 0 \f$
 * 
 * \returns number of found roots
 */
template <typename T>
int poly_solve_quadratic(T const a, T const b, T const c, T *x0, T *x1)
{
    //Handle linear case
    if (a == T{})
    {
        if (b == T{})
        {
            //number of found roots is 0
            return 0;
        }
        else
        {
            *x0 = -c / b;
            //number of found roots is 1
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
            //number of found roots is 2
            return 2;
        }
        else if (disc == T{})
        {
            *x0 = -0.5 * b / a;
            *x1 = -0.5 * b / a;
            //number of found roots is 2
            return 2;
        }
        else
        {
            //number of found roots is 0
            return 0;
        }
    }
}

/*! 
 * Find a minimum in x=[0,1] of the interpolating quadratic through
 * (0,f0) (1,f1) with derivative fp0 at x=0.  The interpolating
 * polynomial is \f$ q(x) = f0 + fp0 * z + (f1-f0-fp0) * z^2 \f$
 */
template <typename T>
T interp_quad(T f0, T fp0, T f1, T zl, T zh)
{
    T fl = f0 + zl * (fp0 + zl * (f1 - f0 - fp0));
    T fh = f0 + zh * (fp0 + zh * (f1 - f0 - fp0));
    //curvature
    T c = 2 * (f1 - f0 - fp0);

    T zmin = zl;
    T fmin = fl;

    if (fh < fmin)
    {
        zmin = zh;
        fmin = fh;
    }
    //positive curvature required for a minimum
    if (c > T{})
    {
        //location of minimum
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
    //Could make an early return by testing curvature >0 for minimum
    T y = cubic<T>(c0, c1, c2, c3, z);

    if (y < *fmin)
    {
        //Accepted new point
        *zmin = z;
        *fmin = y;
    }
}

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
    //found 2 roots
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
    //found 1 root
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
 * recommended values from Fletcher are 
 * rho = 0.01, sigma = 0.1, tau1 = 9, tau2 = 0.05, tau3 = 0.5 
 */
template <typename T, class TFD>
bool minimize(TFD *fn, T rho, T sigma, T tau1, T tau2, T tau3, int order, T alpha1, T *alpha_new)
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

    fn->fdf(0, &f0, &fp0);

    falpha_prev = f0;
    fpalpha_prev = fp0;

    // Avoid uninitialized variables morning
    b = alpha;
    fa = f0;
    fpa = fp0;

    std::size_t i = 0;
    std::size_t const bracket_iters = 100;
    // Begin bracketing
    while (i++ < bracket_iters)
    {
        //TO CHECK
        falpha = fn->f(alpha);

        // Fletcher's rho test

        if (falpha > f0 + alpha * rho * fp0 || falpha >= falpha_prev)
        {
            a = alpha_prev;
            fa = falpha_prev;
            fpa = fpalpha_prev;
            b = alpha;
            fb = falpha;
            fpb = std::numeric_limits<T>::quiet_NaN();
            //goto sectioning
            break;
        }

        fpalpha = fn->df(alpha);

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
            //goto sectioning
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

    //Sectioning of bracket [a,b]
    std::size_t const section_iters = 100;
    while (i++ < section_iters)
    {
        delta = b - a;

        {
            T lower = a + tau2 * delta;
            T upper = b - tau3 * delta;

            alpha = interpolate<T>(a, fa, fpa, b, fb, fpb, lower, upper, order);
        }

        falpha = fn->f(alpha);

        if ((a - alpha) * fpa <= std::numeric_limits<T>::epsilon())
        {
            //Roundoff prevents progress
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " The minimizer is unable to improve on its current estimate, either due" << std::endl;
            std::cerr << " to numerical difficulty or because a genuine local minimum has been reached." << std::endl;
            return false;
        };

        if (falpha > f0 + rho * alpha * fp0 || falpha >= fa)
        {
            //\f$ a_next = a \f$
            b = alpha;
            fb = falpha;
            fpb = std::numeric_limits<T>::quiet_NaN();
        }
        else
        {
            fpalpha = fn->df(alpha);

            if (std::abs(fpalpha) <= -sigma * fp0)
            {
                *alpha_new = alpha;
                //terminate
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

#endif
