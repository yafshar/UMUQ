#ifndef UMHBM_MULTIMIN_LINEAR_WRAPPER_H
#define UMHBM_MULTIMIN_LINEAR_WRAPPER_H

/*! \class function_fdf
 *  \ingroup multimin_Module
 * 
 * \brief Definition of an arbitrary differentiable function with vector input and parameters
 *  
 * \tparam T      data type
 * \tparan TMFD   multimin differentiable function type
 */
template <typename T, class TMFD>
class wrapper_t
{
  public:
    wrapper_t() : fdf_linear(*this) {}

    /*!
     * \brief 
     * 
     */
    void prepare(TMFD *fdf, T const *x_, T const f_, T const *g_, T const *p_, T *x_alpha_, T *g_alpha_)
    {
        wfdf = fdf;
        n = wfdf->n;
        fdf_linear.n = n;

        x = x_;
        f_alpha = f_;
        g = g_;
        p = p_;
        x_alpha = x_alpha_;
        g_alpha = g_alpha_;
        x_cache_key = T{};
        f_cache_key = T{};
        g_cache_key = T{};
        df_cache_key = T{};

        std::copy(x, x + n, x_alpha);
        std::copy(g, g + n, g_alpha);

        df_alpha = slope();
    }

    /*!
     * \brief 
     * 
     */
    void update_position(T alpha, T *x_, T *f, T *g_)
    {
        //Ensure that everything is fully cached
        {
            T f_alpha_;
            T df_alpha_;

            fdf_linear.fdf(alpha, &f_alpha_, &df_alpha_);
        };

        *f = f_alpha;

        std::copy(x_alpha, x_alpha + n, x_);
        std::copy(g_alpha, g_alpha + n, g_);
    }

    /*!
     * \brief 
     * 
     */
    void change_direction()
    {
        //Convert the cache values from the end of the current minimization
        //to those needed for the start of the next minimization, alpha=0

        //The new x_alpha for alpha=0 is the current position
        std::copy(x, x + n, x_alpha);

        x_cache_key = T{};

        //The function value does not change
        f_cache_key = T{};

        //The new g_alpha for alpha=0 is the current gradient at the endpoint
        std::copy(g, g + n, g_alpha);

        g_cache_key = T{};

        //Calculate the slope along the new direction vector, p
        df_alpha = slope();

        df_cache_key = T{};
    }

    /*!
     * \brief 
     * 
     */
    void moveto(T const alpha)
    {
        //using previously cached position
        if (alpha == x_cache_key)
        {
            return;
        }

        //Set \f$ x_alpha = x + alpha * p \f$
        std::copy(x, x + n, x_alpha);

        for (std::size_t i = 0; i < n; i++)
        {
            x_alpha[i] += alpha * p[i];
        }

        x_cache_key = alpha;
    }

    /*!
	 * \brief compute gradient . direction
	 */
    inline T slope()
    {
        T df(0);
        for (std::size_t i = 0; i < n; i++)
        {
            df += g_alpha[i] * p[i];
        }

        return df;
    }

    class wrap : public function_fdf<T, wrap>
    {
      public:
        /*!
         * \brief 
         * 
         */
        wrap(wrapper_t<T, TMFD> &wrapper_t_ref) : w(wrapper_t_ref) {}

        /*!
         * \brief 
         * 
         */
        T f(T const alpha)
        {
            //using previously cached f(alpha)
            if (alpha == w.f_cache_key)
            {
                return w.f_alpha;
            }

            w.moveto(alpha);

            w.f_alpha = w.wfdf->f(w.x_alpha);

            w.f_cache_key = alpha;

            return w.f_alpha;
        }

        /*!
         * \brief
         *
         */
        T df(T const alpha)
        {
            //using previously cached df(alpha)
            if (alpha == w.df_cache_key)
            {
                return w.df_alpha;
            }

            w.moveto(alpha);

            if (alpha != w.g_cache_key)
            {
                w.wfdf->df(w.x_alpha, w.g_alpha);

                w.g_cache_key = alpha;
            }

            w.df_alpha = w.slope();

            w.df_cache_key = alpha;

            return w.df_alpha;
        }

        /*!
         * \brief
         *
         */
        void fdf(T const alpha, T *f, T *df)
        {
            //Check for previously cached values
            if (alpha == w.f_cache_key && alpha == w.df_cache_key)
            {
                *f = w.f_alpha;
                *df = w.df_alpha;
                return;
            }

            if (alpha == w.f_cache_key || alpha == w.df_cache_key)
            {
                *f = w.fdf_linear.f(alpha);
                *df = w.fdf_linear.df(alpha);
                return;
            }

            w.moveto(alpha);

            w.wfdf->fdf(w.x_alpha, &w.f_alpha, w.g_alpha);

            w.f_cache_key = alpha;
            w.g_cache_key = alpha;

            w.df_alpha = w.slope();
            w.df_cache_key = alpha;

            *f = w.f_alpha;
            *df = w.df_alpha;
        }

      private:
        wrapper_t<T, TMFD> &w;
    };

  private:
    TMFD *wfdf;

    //Fixed values
    T const *x;
    T const *g;
    T const *p;

    //Cached values, for x(alpha) = x + alpha * p
    T f_alpha;
    T df_alpha;
    T *x_alpha;
    T *g_alpha;

    //Cache "keys"
    T f_cache_key;
    T df_cache_key;
    T x_cache_key;
    T g_cache_key;

    std::size_t n;

  public:
    wrap fdf_linear;
};

#endif
