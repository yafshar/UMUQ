#ifndef UMHBM_MULTIMIN_H
#define UMHBM_MULTIMIN_H

#define MULTIMIN_FN_EVAL(F, x) (*((F)->f))(x, (F)->params)
#define MULTIMIN_FN_EVAL_F(F, x) (*((F)->f))(x, (F)->params)
#define MULTIMIN_FN_EVAL_DF(F, x, g) (*((F)->df))(x, (F)->params, (g))
#define MULTIMIN_FN_EVAL_F_DF(F, x, y, g) (*((F)->fdf))(x, (F)->params, (y), (g))

/*! \class function_fdf
  * \brief Definition of an arbitrary differentiable function
  *  
  * \tparam T   data type
  * \tparan TFD differentiable function type
  */
template <typename T, class TFD>
class function_fdf
{
  public:
    T f(T const x)
    {
        return static_cast<TFD *>(this)->f(x);
    }

    T df(T const x)
    {
        return static_cast<TFD *>(this)->df(x, df_);
    }

    void fdf(T const x, T *f_, T *df_)
    {
        static_cast<TFD *>(this)->fdf(x, f_, df_);
    }

    function_fdf(size_t n_) : n(n_) {}

    size_t n;

  private:
    friend TFD;
};

// The goal is finding minima of arbitrary multidimensional functions.

/*! \class multimin_function
  * \brief Definition of an arbitrary function with vector input and parameters
  * 
  * \tparam T     data type
  * \tparan TMF   multimin function type
  */
template <typename T, class TMF>
class multimin_function
{
  public:
    T f(T const *x)
    {
        return static_cast<TMF *>(this)->f(x);
    }

    multimin_function(size_t n_) : n(n_) {}

    size_t n;

  private:
    friend TMF;
};

/*! \class multimin_fminimizer_type
  * \brief minimization of non-differentiable functions
  * 
  * \tparam T     data type
  * \tparam TMFMT multimin function minimizer type
  * \tparan TMF   multimin function type
  */
template <typename T, class TMFMT, class TMF>
class multimin_fminimizer_type
{
  public:
    const char *name;

    bool alloc(size_t n)
    {
        return static_cast<TMFMT *>(this)->alloc(n);
    }

    bool set(TMF *tmf, T const *x, T *size, T const *step_size)
    {
        return static_cast<TMFMT *>(this)->set(tmf, x, size, step_size);
    }

    bool iterate(TMF *tmf, T const *x, T *size, T *fval)
    {
        return static_cast<TMFMT *>(this)->iterate(tmf, x, size, fval);
    }

    void free()
    {
        static_cast<TMFMT *>(this)->free();
    }

    multimin_fminimizer_type(const char *name_) : name(name_) {}
    multimin_fminimizer_type(multimin_fminimizer_type const &m) : name(m.name) {}

  private:
    friend TMFMT;
};

/*! \class multimin_fminimizer
  * \brief This class is for minimizing functions without derivatives.
  * 
  * \tparam T      data type
  * \tparam TMFDMT multimin differentiable function minimizer type
  * \tparan TMFD   multimin differentiable function type
  */
template <typename T, class TMFM, class TMFMT, class TMF>
class multimin_fminimizer
{
  public:
    //Multi dimensional part
    TMFMT type;
    TMF f;
    T fval;
    T *x;
    T size;
};

/*! \class multimin_function_fdf
  * \brief Definition of an arbitrary differentiable function with vector input and parameters
  *  
  * \tparam T      data type
  * \tparan TMFD   multimin differentiable function type
  */
template <typename T, class TMFD>
class multimin_function_fdf
{
  public:
    T f(T const *x)
    {
        return static_cast<TMFD *>(this)->f(x);
    }

    T df(T const *x, T *df_)
    {
        return static_cast<TMFD *>(this)->df(x, df_);
    }

    T fdf(T const *x, T *f_, T *df_)
    {
        return static_cast<TMFD *>(this)->fdf(x, f_, df_);
    }

    multimin_function_fdf(size_t n_) : n(n_) {}

    size_t n;

  private:
    friend TMFD;
};

/*! \class multimin_fdfminimizer_type
  * \brief differentiable function minimizer type
  * 
  * \tparam T      data type
  * \tparam TMFDMT multimin differentiable function minimizer type
  * \tparan TMFD   multimin differentiable function type
  */
template <typename T, class TMFDMT, class TMFD>
class multimin_fdfminimizer_type
{
  public:
    const char *name;

    bool alloc(size_t n)
    {
        return static_cast<TMFDMT *>(this)->alloc(n);
    }

    bool set(TMFD *tmfd, T const *x, T *f, T *gradient, T step_size, T tol)
    {
        return static_cast<TMFDMT *>(this)->set(tmfd, x, f, gradient, step_size, tol);
    }

    bool iterate(TMFD *tmfd, T *x, T *f, T *gradient, T *dx);
    {
        return static_cast<TMFDMT *>(this)->iterate(tmfd, x, f, gradient, dx);
    }

    bool restart()
    {
        return static_cast<TMFDMT *>(this)->restart();
    }

    void free()
    {
        static_cast<TMFDMT *>(this)->free();
    }

    multimin_fdfminimizer_type(const char *name_) : name(name_) {}
    multimin_fdfminimizer_type(multimin_fdfminimizer_type const &m) : name(m.name) {}

  private:
    friend TMFDMT;
};

/*! \class multimin_fdfminimizer
  * \brief This class is for minimizing functions using derivatives. 
  * 
  * \tparam T      data type
  * \tparam TMFDMT multimin differentiable function minimizer type
  * \tparan TMFD   multimin differentiable function type
  */
template <typename T, class TMFDMT, class TMFD>
class multimin_fdfminimizer
{
  public:
    /*!
     * \brief alloc
     * 
     * \param Ttype pointer to \a multimin_fdfminimizer_type object 
     * \param n_ size of array
     * 
     * \returns true if everything goes OK
     */
    bool alloc(multimin_fdfminimizer_type<T, TMFDMT, TMFD> *Ttype, size_t n_)
    {
        n = n_;
        type = Ttype;

        try
        {
            x = new T[n];
            //set to zero
            gradient = new T[n]();
            //set to zero
            dx = new T[n]();
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }

        if (!type->alloc(n))
        {
            free();
            return false;
        }

        return true;
    }

    /*!
     * \brief set
     * 
     * \param mfdf      pointer to an arbitrary differentiable real-valued function
     * \param x         input array
     * \param n_        size of array n_  
     * \param step_size step size
     * \param tol       tol
     *  
     * returns true if everything goes OK
     */
    bool set(multimin_function_fdf<T, TMFD> *mfdf, T const *x_, size_t n_, T step_size, T tol)
    {
        if (n != mfdf->n)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Function incompatible with solver size! " << std::endl;
            return false;
        }

        if (n_ != mfdf->n)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Vector length not compatible with function" << std::endl;
            return false;
        }

        //set the pointer
        fdf = mfdf;

        //copy array x_ to array x
        std::copy(x_, x_ + n_, x);

        //set dx to zero
        std::fill(dx, dx + n, (T)0);

        return type->set(fdf, x, &f, gradient, step_size, tol);
    }

    /*!
     * \brief iterate
     * 
     */
    bool iterate()
    {
        return type->iterate(fdf, x, &f, gradient, dx);
    }

    /*!
     * \brief restart
     * 
     */
    bool restart()
    {
        return type->restart();
    }

    /*!
     * \brief name
     * \returns the name of the minimization type
     */
    const char *name()
    {
        return type->name;
    }

    /*!
     * \brief minimum
     * \returns the minimum
     */
    T minimum()
    {
        return f;
    }

    /*!
     * \brief get function x
     * \returns x
     */
    T *get_x()
    {
        return x;
    }

    /*!
     * \brief get function dx
     * \returns dx
     */
    T *get_dx()
    {
        return dx;
    }

    /*!
     * \brief get function x
     * \returns x
     */
    T *get_gradient()
    {
        return gradient;
    }

    /*!
     * \brief destructor
     * 
     */
    void free()
    {
        type->free();
        type = nullptr;

        fdf = nullptr;

        delete[] x;
        x = nullptr;

        delete[] gradient;
        gradient = nullptr;

        delete[] dx;
        dx = nullptr;

        n = 0;
        f = 0;
    }

  private:
    // multi dimensional part
    multimin_fdfminimizer_type<T, TMFDMT, TMFD> *type;
    multimin_function_fdf<T, TMFD> *fdf;

    T f;
    
    T *x;
    T *gradient;
    T *dx;

    size_t n;
};

/*! \class multimin
* \brief 
*	
*/
struct multimin
{
};

template <typename T>
int multimin_test_gradient(T const *g, size_t const n, T const epsabs)
{
    if (epsabs < (T)0)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Absolute tolerance is negative" << std::endl;
        //fail
        return -1;
    }

    //First compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
    T norm(0);
    std::for_each(g, g + n, [&](T const g_) { norm += g_ * g_; });
    if (std::sqrt(norm) < epsabs)
    {
        //success
        return 0;
    }

    //continue
    return 1;
}

template <typename T>
int multimin_test_size(T const size, T const epsabs)
{
    if (epsabs < 0)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Absolute tolerance is negative" << std::endl;
        //fail
        return -1;
    }

    if (size < epsabs)
    {
        //success
        return 0;
    }

    //continue
    return 1;
}

template <typename T, class TMF>
bool multimin_diff(TMF const *f, T const *x, T *g)
{
    size_t n = f->n;

    T const h = std::sqrt(std::numeric_limits<T>::epsilon());

    T *x1 = nullptr;

    try
    {
        x1 = new T[n];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        return false;
    }

    std::copy(x, x + n, x1);

    for (size_t i = 0; i < n; i++)
    {
        T fl;
        T fh;

        T xi = x[i];

        T dx = std::abs(xi) * h;
        if (dx <= 0.0)
        {
            dx = h;
        }

        x1[i] = xi + dx;

        fh = f->f(x1);

        x1[i] = xi - dx;

        fl = f->f(x1);

        x1[i] = xi;

        g[i] = (fh - fl) / ((T)2 * dx);
    }

    delete[] x1;

    return true;
}

#endif
