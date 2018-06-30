#ifndef UMUQ_MULTIMIN_STEEPEST_DESCENT_H
#define UMUQ_MULTIMIN_STEEPEST_DESCENT_H

/*! \class steepest_descent
 *  \ingroup multimin_Module
 * 
 * \brief steepest_descent for differentiable function minimizer type
 * 
 * The steepest descent algorithm follows the downhill gradient of the function at each step. 
 * When a downhill step is successful the step-size is increased by a factor of two. 
 * If the downhill step leads to a higher function value then the algorithm backtracks 
 * and the step size is decreased using the parameter tol. 
 * 
 * A suitable value of tol for most applications is 0.1. 
 * The steepest descent method is inefficient and is included only for demonstration purposes. 
 * 
 * \tparam T      data type
 * \tparam TMFD   multimin differentiable function type
 */
template <typename T, class TMFD>
class steepest_descent : public multimin_fdfminimizer_type<T, steepest_descent<T, TMFD>, TMFD>
{
  public:
    /*!
     * \brief constructor
     * 
     * \param name name of the differentiable function minimizer type (default "steepest_descent")
     */
    steepest_descent(const char *name_ = "steepest_descent") : x1(nullptr),
                                                               g1(nullptr) { this->name = name_; }
    /*!
     * \brief destructor
     */
    ~steepest_descent() { free(); }

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
            x1 = new T[n_]();
            g1 = new T[n_]();
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
    bool set(TMFD *fdf, T const *x, T *f, T *gradient, T step_size, T tol_)
    {
        fdf->fdf(x, f, gradient);

        step = step_size;
        max_step = step_size;
        tol = tol_;

        return true;
    }

    void free()
    {
        if (x1 != nullptr)
        {
            delete[] x1;
            x1 = nullptr;
        }

        if (g1 != nullptr)
        {
            delete[] g1;
            g1 = nullptr;
        }
    }

    bool restart()
    {
        step = max_step;
        return true;
    }

    bool iterate(TMFD *fdf, T *x, T *f, T *gradient, T *dx)
    {
        T f0 = *f;

        bool failed(false);

        //Compute new trial point at x1= x - step * dir, where dir is the normalized gradient

        //First compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
        T gnorm(0);
        std::for_each(gradient, gradient + n, [&](T const g) { gnorm += g * g; });
        if (gnorm <= T{})
        {
            //set dx to zero
            std::fill(dx, dx + n, T{});

			UMUQFAILRETURN("The minimizer is unable to improve on its current estimate, either due \n to the numerical difficulty or because a genuine local minimum has been reached!");
        }

        gnorm = std::sqrt(gnorm);

        T f1 = 2 * f0;
        while (f1 > f0)
        {
            //Compute the sum \f$y = \alpha x + y\f$ for the vectors x and y.
            //(set dx to zero)
            T const alpha = -step / gnorm;
            for (std::size_t i = 0; i < n; i++)
            {
                dx[i] = alpha * gradient[i];
            }

            std::copy(x, x + n, x1);

            for (std::size_t i = 0; i < n; i++)
            {
                x1[i] += dx[i];
            }

            //Evaluate function and gradient at new point x1
            fdf->fdf(x1, &f1, g1);

            if (f1 > f0)
            {
                //Downhill step failed, reduce step-size and try again
                failed = true;
                step *= tol;
            }
        }

        failed ? step *= tol : step *= 2;

        std::copy(x1, x1 + n, x);
        std::copy(g1, g1 + n, gradient);

        *f = f1;

        return true;
    }

  private:
    T step;
    T max_step;
    T tol;

    T *x1;
    T *g1;

    std::size_t n;
};

#endif
