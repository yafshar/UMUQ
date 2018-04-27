#ifndef UMHBM_MULTIMIN_DIRECTIONAL_MINIMIZE_H
#define UMHBM_MULTIMIN_DIRECTIONAL_MINIMIZE_H

template <typename T>
inline void take_step(size_t n, T const *x, T const *p, T step, T lambda, T *x1, T *dx)
{
    //Compute the sum \f$y = \alpha x + y\f$ for the vectors x and y.
    //(set dx to zero)
    T alpha = -step / lambda;
    for (size_t i = 0; i < n; i++)
    {
        dx[i] = alpha * p[i];
    }
    std::copy(x, x + n, x1);
    for (size_t i = 0; i < n; i++)
    {
        x1[i] += dx[i];
    }
}

template <typename T, class TMFD>
void intermediate_point(TMFD *fdf, T const *x, T const *p, T lambda, T pg, T stepa, T stepc, T fa, T fc, T *x1, T *dx, T *gradient, T *step, T *f)
{
    T stepb(1);
    T fb(fa);

    size_t n = fdf->n;

    while (fb >= fa && stepb > 0.0)
    {
        T u = std::abs(pg * lambda * stepc);
        stepb = 0.5 * stepc * u / ((fc - fa) + u);

        take_step<T>(n, x, p, stepb, lambda, x1, dx);

        fb = fdf->f(x1);

#ifdef DEBUG
        std::cout << "Trying stepb = " << stepb << " fb = " << fb << std::endl;
#endif

        if (fb >= fa && stepb > 0.0)
        {
            //Downhill step failed, reduce step-size and try again
            fc = fb;
            stepc = stepb;
        }
    }

#ifdef DEBUG
    std::cout << "Ok!" << std::endl;
#endif

    *step = stepb;
    *f = fb;
    fdf->df(x1, gradient);
}

template <typename T, class TMFD>
void minimize(TMFD *fdf, T const *x, T const *p, T lambda, T stepa, T stepb, T stepc, T fa, T fb, T fc, T tol, T *x1, T *dx1, T *x2, T *dx2, T *gradient, T *step, T *f, T *gnorm)
{
    // Starting at (x0, f0) move along the direction p to find a minimum
    // f(x0 - lambda * p), returning the new point x1 = x0-lambda*p,
    // f1=f(x1) and g1 = grad(f) at x1

    T u = stepb;
    T v = stepa;
    T w = stepc;
    T fu = fb;
    T fv = fa;
    T fw = fc;

    T old2 = std::abs(w - v);
    T old1 = std::abs(v - u);

    T stepm;
    T fm;
    T pg;
    T gnorm1;

    size_t n = fdf->n;

    std::copy(x1, x1 + n, x2);
    std::copy(dx1, dx1 + n, dx2);

    *f = fb;
    *step = stepb;

    T s(0);
    std::for_each(gradient, gradient + n, [&](T const g) { s += g * g; });

    *gnorm = std::sqrt(s);

    int iter(0);

mid_trial:

    iter++;

    if (iter > 10)
    {
        return; // MAX ITERATIONS
    }

    {
        T dw = w - u;
        T dv = v - u;
        T du = 0.0;

        T e1 = ((fv - fu) * dw * dw + (fu - fw) * dv * dv);
        T e2 = 2.0 * ((fv - fu) * dw + (fu - fw) * dv);

        if (e2 != 0.0)
        {
            du = e1 / e2;
        }

        if (du > 0.0 && du < (stepc - stepb) && std::abs(du) < 0.5 * old2)
        {
            stepm = u + du;
        }
        else if (du < 0.0 && du > (stepa - stepb) && std::abs(du) < 0.5 * old2)
        {
            stepm = u + du;
        }
        else if ((stepc - stepb) > (stepb - stepa))
        {
            stepm = 0.38 * (stepc - stepb) + stepb;
        }
        else
        {
            stepm = stepb - 0.38 * (stepb - stepa);
        }
    }

    take_step<T>(n, x, p, stepm, lambda, x1, dx1);

    fm = fdf->f(x1);

#ifdef DEBUG
    std::cout << "Trying stepm = " << stepm << " fm = " << fm << std::endl;
#endif

    if (fm > fb)
    {
        if (fm < fv)
        {
            w = v;
            v = stepm;
            fw = fv;
            fv = fm;
        }
        else if (fm < fw)
        {
            w = stepm;
            fw = fm;
        }

        if (stepm < stepb)
        {
            stepa = stepm;
            fa = fm;
        }
        else
        {
            stepc = stepm;
            fc = fm;
        }
        goto mid_trial;
    }
    else if (fm <= fb)
    {
        old2 = old1;
        old1 = std::abs(u - stepm);
        w = v;
        v = u;
        u = stepm;
        fw = fv;
        fv = fu;
        fu = fm;

        std::copy(x1, x1 + n, x2);
        std::copy(dx1, dx1 + n, dx2);

        fdf->df(x1, gradient);

        pg = (T)0;
        for (size_t i = 0; i < n; i++)
        {
            pg += p[i] * gradient[i];
        }

        s = (T)0;
        std::for_each(gradient, gradient + n, [&](T const g) { s += g * g; });
        gnorm1 = std::sqrt(s);

#ifdef DEBUG
        //TODO Use IO class to print out p, gradient, pg for debugging purpose
#endif
        *f = fm;
        *step = stepm;
        *gnorm = gnorm1;

        if (std::abs(pg * lambda / gnorm1) < tol)
        {
#ifdef DEBUG
            std::cout << "Ok!" << std::endl;
#endif
            return; // SUCCESS
        }

        if (stepm < stepb)
        {
            stepc = stepb;
            fc = fb;
            stepb = stepm;
            fb = fm;
        }
        else
        {
            stepa = stepb;
            fa = fb;
            stepb = stepm;
            fb = fm;
        }
        goto mid_trial;
    }
}

#endif
