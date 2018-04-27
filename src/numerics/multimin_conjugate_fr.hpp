#ifndef UMHBM_MULTIMIN_CONJUGATE_FR_H
#define UMHBM_MULTIMIN_CONJUGATE_FR_H

#include "multimin.hpp"
#include "multimin_directional_minimize.hpp"

/*! \class conjugate_fr
  * \brief Conjugate gradient Fletcher-Reeve algorithm
  * 
  * \tparam T      data type
  * \tparan TMFD   multimin differentiable function type
  */
template <typename T, class TMFD>
class conjugate_fr : public multimin_fdfminimizer_type<T, conjugate_fr<T, TMFD>, TMFD>
{
  public:
    /*!
     * \brief constructor
     * 
     * \param name name of the differentiable function minimizer type (default "conjugate_fr")
     */
    conjugate_fr(const char *name_ = "conjugate_fr";) : name(name_),
                                                        iter(0),
                                                        step(0),
                                                        max_step(0),
                                                        tol(0),
                                                        x1(nullptr),
                                                        dx1(nullptr),
                                                        x2(nullptr),
                                                        pnorm(0),
                                                        p(nullptr),
                                                        g0norm(0),
                                                        g0(nullptr),
                                                        n(0) {}

    /*!
     * \brief destructor
     */
    ~conjugate_fr() { free(); }

    /*!
     * \brief allocate space for data type T
     * 
     * \param n_ size of array
     * 
     * \returns false if there is insufficient memory to create data array 
     */
    bool alloc(size_t n_)
    {
        n = n_;
        try
        {
            x1 = new T[n]();
            dx1 = new T[n]();
            x2 = new T[n]();
            p = new T[n]();
            g0 = new T[n]();
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

        // Use the gradient as the initial direction
        std::copy(gradient, gradient + n, p);
        std::copy(gradient, gradient + n, g0);

        {
            //First compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
            T gnorm(0);
            std::for_each(gradient, gradient + n, [&](T const g) { gnorm += g * g; });

            pnorm = std::sqrt(gnorm);
            g0norm = pnorm;
        }

        return true;
    }

    void free()
    {
        delete[] x1;
        x1 = nullptr;
        delete[] dx1;
        dx1 = nullptr;
        delete[] x2;
        x2 = nullptr;
        delete[] p;
        p = nullptr;
        delete[] g0;
        g0 = nullptr;
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

        if (pnorm <= (T)0 || g0norm <= 0.0)
        {
            //set dx to zero
            std::fill(dx, dx + n, (T)0);

            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " The minimizer is unable to improve on its current estimate, either due" << std::endl;
            std::cerr << " to numerical difficulty or because a genuine local minimum has been reached." << std::endl;
            return false;
        }

        T pg(0);
        //Determine which direction is downhill, +p or -p
        for (size_t i = 0; i < n; i++)
        {
            pg += p[i] * gradient[i];
        }

        dir = (pg >= 0.0) ? +1.0 : -1.0;

        //Compute new trial point at x_c= x - step * p, where p is the current direction
        take_step<T>(n, x, p, stepc, dir / pnorm, x1, dx);

        T fc;
        /* Evaluate function and gradient at new point xc */
        fc = fdf->f(x1);

        if (fc < fa)
        {
            /* Success, reduced the function value */
            step = stepc * 2.0;
            *f = fc;
            std::copy(x1, x1 + n, x);

            fdf->df(x1, gradient);

            return true;
        }

#ifdef DEBUG
        std::cout << "Got stepc = " << stepc << "fc = " << fc << std::endl;
#endif

        // Do a line minimisation in the region (xa,fa) (xc,fc) to find an
        // intermediate (xb,fb) satisifying fa > fb < fc.  Choose an initial
        // xb based on parabolic interpolation
        intermediate_point<T, TMFD>(fdf, x, p, dir / pnorm, pg, stepa, stepc, fa, fc, x1, dx1, gradient, &stepb, &fb);

        if (stepb == 0.0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " The minimizer is unable to improve on its current estimate, either due" << std::endl;
            std::cerr << " to numerical difficulty or because a genuine local minimum has been reached." << std::endl;
            return false;
        }

        minimize<T, TMFD>(fdf, x, p, dir / pnorm, stepa, stepb, stepc, fa, fb, fc, tol, x1, dx1, x2, dx, gradient, &step, f, &g1norm);

        std::copy(x2, x2 + n, x);

        // Choose a new conjugate direction for the next step
        iter = (iter + 1) % n;

        if (iter == 0)
        {
            std::copy(gradient, gradient + n, p);
            pnorm = g1norm;
        }
        else
        {
            //p' = g1 - beta * p
            T beta = std::pow(g1norm / g0norm, 2);

            std::for_each(p, p + n, [&](T &p_) { p_ *= beta; });

            for (size_t i = 0; i < n; i++)
            {
                p[i] += gradient[i];
            }

            //First compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
            T s(0);
            std::for_each(p, p + n, [&](T const p_) { s += p_ * p_; });
            pnorm = std::sqrt(s);
        }

        g0norm = g1norm;
        std::copy(gradient, gradient + n, g0);

#ifdef DEBUG
        std::cout << "updated conjugate directions" << std::endl;
        //TODO print vector p and g
#endif

        return true;
    }

  private:
    int iter;

    T step;
    T max_step;
    T tol;
    T *x1;
    T *dx1;
    T *x2;
    T pnorm;
    T *p;
    T g0norm;
    T *g0;

    size_t n;
}

#endif
