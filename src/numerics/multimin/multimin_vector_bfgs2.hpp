#ifndef UMUQ_MULTIMIN_VECTOR_BFGS2_H
#define UMUQ_MULTIMIN_VECTOR_BFGS2_H

/*! \class vector_bfgs2
 *  \ingroup multimin_Module
 * 
 * \brief Limited memory Broyden-Fletcher-Goldfarb-Shanno method
 * Fletcher's implementation of the BFGS method,
 * using the line minimization algorithm from from R.Fletcher,
 * "Practical Methods of Optimization", Second Edition, ISBN
 * 0471915475.  Algorithms 2.6.2 and 2.6.4.
 * 
 * 
 * \tparam T      data type
 * \tparam TMFD   multimin differentiable function type
 */
template <typename T, class TMFD>
class vector_bfgs2 : public multimin_fdfminimizer_type<T, vector_bfgs2<T, TMFD>, TMFD>
{
  public:
    /*!
     * \brief constructor
     * 
     * \param name name of the differentiable function minimizer type (default "vector_bfgs2")
     */
    vector_bfgs2(const char *name_ = "vector_bfgs2") : p(nullptr),
                                                       x0(nullptr),
                                                       g0(nullptr),
                                                       dx0(nullptr),
                                                       dg0(nullptr),
                                                       x_alpha(nullptr),
                                                       g_alpha(nullptr) { this->name = name_; }

    /*!
     * \brief destructor
     */
    ~vector_bfgs2() { free(); }

    /*!
     * \brief allocate space for data type T
     * 
     * \param n_ size of array
     * 
     * \returns false if there is insufficient memory to create data array 
     */
    bool alloc(std::size_t const n_)
    {
        n = n_;
        try
        {
            p = new T[n]();
            x0 = new T[n]();
            g0 = new T[n]();
            dx0 = new T[n]();
            dg0 = new T[n]();
            x_alpha = new T[n]();
            g_alpha = new T[n]();
        }
        catch (std::bad_alloc &e)
        {
			UMUQFAILRETURN("Failed to allocate memory!");
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
        delta_f = T{};

        fdf->fdf(x, f, gradient);

        // Use the gradient as the initial direction
        std::copy(x, x + n, x0);
        std::copy(gradient, gradient + n, g0);

        {
            //Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
            T s(0);
            std::for_each(g0, g0 + n, [&](T const g_i) { s += g_i * g_i; });

            g0norm = std::sqrt(s);
        }

        std::copy(gradient, gradient + n, p);

        {
            T const alpha = -static_cast<T>(1) / g0norm;
            std::for_each(p, p + n, [&](T &p_i) { p_i *= alpha; });
        }

        {
            //Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
            T s(0);
            std::for_each(p, p + n, [&](T const p_i) { s += p_i * p_i; });

            //should be 1
            pnorm = std::sqrt(s);
        }

        fp0 = -g0norm;

        //Prepare the wrapper
        w.prepare(fdf, x0, *f, g0, p, x_alpha, g_alpha);

        //Prepare 1d minimization parameters
        rho = static_cast<T>(0.01);
        sigma = tol;
        tau1 = static_cast<T>(9);
        tau2 = static_cast<T>(0.05);
        tau3 = static_cast<T>(0.5);

        //Use cubic interpolation where possible
        order = 3;

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
        if (dg0 != nullptr)
        {
            delete[] dg0;
            dg0 = nullptr;
        }
        if (g_alpha != nullptr)
        {
            delete[] g_alpha;
            g_alpha = nullptr;
        }
        if (x_alpha != nullptr)
        {
            delete[] x_alpha;
            x_alpha = nullptr;
        }
    }

    bool restart()
    {
        iter = 0;
        return true;
    }

    bool iterate(TMFD *fdf, T *x, T *f, T *gradient, T *dx)
    {

        if (pnorm == T{} || g0norm == T{} || fp0 == T{})
        {
            //set dx to zero
            std::fill(dx, dx + n, T{});

			UMUQFAILRETURN("The minimizer is unable to improve on its current estimate, either due \n to the numerical difficulty or because a genuine local minimum has been reached!");
        }

        T alpha(0);
        T alpha1;

        T pg;
        T dir;

        T f0 = *f;

        if (delta_f < T{})
        {
            T del = std::max(-delta_f, 10 * std::numeric_limits<T>::epsilon() * std::abs(f0));

            alpha1 = std::min(static_cast<T>(1), -2 * del / fp0);
        }
        else
        {
            alpha1 = std::abs(step);
        }

        //Line minimization, with cubic interpolation (order = 3)
        bool status = minimize<T, function_fdf<T, class wrapper_t<T, TMFD>::wrap>>(&w.fdf_linear, rho, sigma,
                                                                                   tau1, tau2, tau3, order,
                                                                                   alpha1, &alpha);
        if (status != true)
        {
            return false;
        }

        w.update_position(alpha, x, f, gradient);

        delta_f = *f - f0;

        //Choose a new direction for the next step
        {
            //This is the BFGS update:
            //\f$ p' = g1 - A dx - B dg \f$
            //\f$ A = - (1+ dg.dg/dx.dg) B + dg.g/dx.dg \f$
            //\f$ B = dx.g/dx.dg \f$

            //\f$ dx0 = x - x0 \f$
            std::copy(x, x + n, dx0);

            for (std::size_t i = 0; i < n; i++)
            {
                dx0[i] -= x0[i];
            }

            //keep a copy
            std::copy(dx0, dx0 + n, dx);

            // \f$ dg0 = g - g0 \f$
            std::copy(gradient, gradient + n, dg0);

            for (std::size_t i = 0; i < n; i++)
            {
                dg0[i] -= g0[i];
            }

            T dxg(0);
            for (std::size_t i = 0; i < n; i++)
            {
                dxg += dx0[i] * gradient[i];
            }

            T dgg(0);
            for (std::size_t i = 0; i < n; i++)
            {
                dgg += dg0[i] * gradient[i];
            }

            T dxdg(0);
            for (std::size_t i = 0; i < n; i++)
            {
                dxdg += dx0[i] * dg0[i];
            }

            T dgnorm;
            {
                T s(0);
                std::for_each(dg0, dg0 + n, [&](T const d_i) { s += d_i * d_i; });

                dgnorm = std::sqrt(s);
            }

            T A;
            T B;

            if (dxdg != T{})
            {
                B = dxg / dxdg;
                A = -(static_cast<T>(1) + dgnorm * dgnorm / dxdg) * B + dgg / dxdg;
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
        }

        std::copy(gradient, gradient + n, g0);
        std::copy(x, x + n, x0);

        {
            T s(0);
            std::for_each(g0, g0 + n, [&](T const g_i) { s += g_i * g_i; });

            g0norm = std::sqrt(s);
        }

        {
            T s(0);
            std::for_each(p, p + n, [&](T const p_i) { s += p_i * p_i; });

            pnorm = std::sqrt(s);
        }

        //Update direction and fp0
        pg = T{};
        for (std::size_t i = 0; i < n; i++)
        {
            pg += p[i] * gradient[i];
        }

        dir = (pg >= T{}) ? -static_cast<T>(1) : static_cast<T>(1);

        {
            T const alpha = dir / pnorm;
            std::for_each(p, p + n, [&](T &p_i) { p_i *= alpha; });
        }

        {
            T s(0);
            std::for_each(p, p + n, [&](T const p_i) { s += p_i * p_i; });

            pnorm = std::sqrt(s);
        }

        fp0 = T{};
        for (std::size_t i = 0; i < n; i++)
        {
            fp0 += g0[i] * p[i];
        }

        w.change_direction();

        return true;
    }

  private:
    int iter;

    T step;

    T pnorm;
    T g0norm;
    T delta_f;

    //f'(0) for f(x-alpha*p)
    T fp0;

    T *p;
    T *x0;
    T *g0;

    //work space
    T *dx0;
    T *dg0;
    T *x_alpha;
    T *g_alpha;

    //wrapper function
    wrapper_t<T, TMFD> w;

    //minimization parameters
    T rho;
    T sigma;
    T tau1;
    T tau2;
    T tau3;

    int order;

    std::size_t n;
};

#endif
