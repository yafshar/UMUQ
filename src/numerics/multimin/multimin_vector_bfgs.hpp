#ifndef UMHBM_MULTIMIN_VECTOR_BFGS_H
#define UMHBM_MULTIMIN_VECTOR_BFGS_H

#include "multimin.hpp"
#include "multimin_directional_minimize.hpp"

/*! \class vector_bfgs
  * \brief Limited memory Broyden-Fletcher-Goldfarb-Shanno method
  * 
  * \tparam T      data type
  * \tparan TMFD   multimin differentiable function type
  */
template <typename T, class TMFD>
class vector_bfgs : public multimin_fdfminimizer_type<T, vector_bfgs<T, TMFD>, TMFD>
{
  public:
    /*!
     * \brief constructor
     * 
     * \param name name of the differentiable function minimizer type (default "vector_bfgs")
     */
    vector_bfgs(const char *name_ = "vector_bfgs") : name(name_),
                                                     p(nullptr),
                                                     x0(nullptr),
                                                     x1(nullptr),
                                                     x2(nullptr),
                                                     g0(nullptr),
                                                     dx0(nullptr),
                                                     dx1(nullptr) {}

    /*!
     * \brief destructor
     */
    ~vector_bfgs() { free(); }

    /*!
     * \brief allocate space for data type T
     * 
     * \param n_ size of array
     * 
     * \returns false if there is insufficient memory to create data array 
     */
    bool alloc(std::size_t n_)
    {
        n = n_;
        try
        {
            p = new T[n]();
            x0 = new T[n]();
            x1 = new T[n]();
            x2 = new T[n]();
            g0 = new T[n]();
            dx0 = new T[n]();
            dx1 = new T[n]();
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    /*!
     * \brief set
     * 
     * \param fdf differentiable function minimizer
     * \param x   array of data
     * \param f   
     * \param gradient
     * \param step_size
     * \param tol
     */
    bool set(TMFD *fdf, T const *x, T *f, T *gradient, T step_size, T tol)
    {
        iter = 0;

        step = step_size;
        max_step = step_size;
        tol = tol;

        fdf->fdf(x, f, gradient);

        //Use the gradient as the initial direction
        std::copy(x, x + n, x0);
        std::copy(gradient, gradient + n, p);
        std::copy(gradient, gradient + n, g0);

        {
            //Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
            T s(0);
            std::for_each(gradient, gradient + n, [&](T const g_i) { s += g_i * g_i; });

            pnorm = std::sqrt(s);
            g0norm = pnorm;
        }

        return true;
    }

    void free()
    {
        if (p != nullptr)
        {
            delete[] p;
            p = nullptr;
        }
        if (x0 != nullptr)
        {
            delete[] x0;
            x0 = nullptr;
        }
        if (x1 != nullptr)
        {
            delete[] x1;
            x1 = nullptr;
        }
        if (x2 != nullptr)
        {
            delete[] x2;
            x2 = nullptr;
        }
        if (g0 != nullptr)
        {
            delete[] g0;
            g0 = nullptr;
        }
        if (dx0 != nullptr)
        {
            delete[] dx0;
            dx0 = nullptr;
        }
        if (dx1 != nullptr)
        {
            delete[] dx1;
            dx1 = nullptr;
        }
    }

    bool restart()
    {
        iter = 0;
        return true;
    }

    bool iterate(TMFD *fdf, T *x, T *f, T *gradient, T *dx)
    {

        T fa = *f;
        T fb;
        T dir;
        T stepa(0);
        T stepb;
        T stepc = step;
        T g1norm;

        if (pnorm <= T{} || g0norm <= T{})
        {
            //Set dx to zero
            std::fill(dx, dx + n, T{});

            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " The minimizer is unable to improve on its current estimate, either due" << std::endl;
            std::cerr << " to numerical difficulty or because a genuine local minimum has been reached." << std::endl;
            return false;
        }

        T pg(0);
        //Determine which direction is downhill, +p or -p
        for (std::size_t i = 0; i < n; i++)
        {
            pg += p[i] * gradient[i];
        }

        dir = (pg >= T{}) ? +1 : -1;

        //Compute new trial point at \f$ x_c= x - step * p, \f$ where p is the current direction
        take_step<T>(n, x, p, stepc, dir / pnorm, x1, dx);

        T fc;
        //Evaluate function and gradient at new point xc
        fc = fdf->f(x1);

        if (fc < fa)
        {
            //Success, reduced the function value
            step = stepc * 2;

            *f = fc;

            std::copy(x1, x1 + n, x);

            fdf->df(x1, gradient);

            return true;
        }

#ifdef DEBUG
        std::cout << "Got stepc = " << stepc << "fc = " << fc << std::endl;
#endif

        //Do a line minimisation in the region (xa,fa) (xc,fc) to find an
        //intermediate (xb,fb) satisifying fa > fb < fc.  Choose an initial
        //xb based on parabolic interpolation
        intermediate_point<T, TMFD>(fdf, x, p, dir / pnorm, pg, stepa, stepc, fa, fc, x1, dx1, gradient, &stepb, &fb);

        if (stepb == T{})
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " The minimizer is unable to improve on its current estimate, either due" << std::endl;
            std::cerr << " to numerical difficulty or because a genuine local minimum has been reached." << std::endl;
            return false;
        }

        minimize<T, TMFD>(fdf, x, p, dir / pnorm, stepa, stepb, stepc, fa, fb, fc, tol, x1, dx1, x2, dx, gradient, &step, f, &g1norm);

        std::copy(x2, x2 + n, x);

        //Choose a new conjugate direction for the next step
        iter = (iter + 1) % n;

        if (iter == 0)
        {
            std::copy(gradient, gradient + n, p);
            pnorm = g1norm;
        }
        else
        {
            //This is the BFGS update:
            //\f$ p' = g1 - A dx - B dg \f$
            //\f$ A = - (1+ dg.dg/dx.dg) B + dg.g/dx.dg \f$
            //\f$ B = dx.g/dx.dg \f$
            T dxg;
            T dgg;
            T dxdg;
            T dgnorm;
            T A;
            T B;

            //\f$ dx0 = x - x0 \f$
            std::copy(x, x + n, dx0);

            for (std::size_t i = 0; i < n; i++)
            {
                dx0[i] -= x0[i];
            }

            //\f$ dg0 = g - g0 \f$
            std::copy(gradient, gradient + n, dg0);

            for (std::size_t i = 0; i < n; i++)
            {
                dg0[i] -= g0[i];
            }

            dxg = T{};
            for (std::size_t i = 0; i < n; i++)
            {
                dxg += dx0[i] * gradient[i];
            }

            dgg = T{};
            for (std::size_t i = 0; i < n; i++)
            {
                dgg += dg0[i] * gradient[i];
            }

            dxdg = T{};
            for (std::size_t i = 0; i < n; i++)
            {
                dxdg += dx0[i] * dg0[i];
            }

            {
                T s(0);
                std::for_each(dg0, dg0 + n, [&](T const d_i) { s += d_i * d_i; });

                dgnorm = std::sqrt(s);
            }

            if (dxdg != T{})
            {
                B = dxg / dxdg;
                A = -(1 + dgnorm * dgnorm / dxdg) * B + dgg / dxdg;
            }
            else
            {
                B = T{};
                A = T{};
            }

            std::copy(gradient, gradient + n, p);

            for (std::size_t i = 0; i < n; i++)
            {
                p[i] -= A * dx0[i];
            }

            for (std::size_t i = 0; i < n; i++)
            {
                p[i] -= B * dg0[i];
            }

            {
                T s(0);
                std::for_each(p, p + n, [&](T const p_i) { s += p_i * p_i; });

                pnorm = std::sqrt(s);
            }
        }

        std::copy(gradient, gradient + n, g0);

        std::copy(x, x + n, x0);

        {
            T s(0);
            std::for_each(g0, g0 + n, [&](T const g_i) { s += g_i * g_i; });

            g0norm = std::sqrt(s);
        }

#ifdef DEBUG
        std::cout << "updated directions" << std::endl;
        //TODO print vector p and g
#endif

        return true;
    }

  private:
    int iter;

    T step;
    T max_step;
    T tol;

    T pnorm;
    T g0norm;

    T *p;
    T *x0;
    T *x1;
    T *x2;
    T *g0;
    T *dx0;
    T *dx1;
    T *dg0;

    std::size_t n;
};

#endif
