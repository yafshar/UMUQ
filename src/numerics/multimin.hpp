#ifndef UMHBM_MULTIMIN_H
#define UMHBM_MULTIMIN_H

/*! \class multimin_function
  * \brief Definition of an arbitrary real-valued function with vector input and parameters
  */
template <typename T>
struct multimin_function
{
    T(*f)
    (T const *x, void *params);
    size_t n;
    void *params;
};

/*! \class multimin_function_fdf
  * \brief Definition of an arbitrary differentiable real-valued function with vector input and parameters
  */
template <typename T>
struct multimin_function_fdf
{
    T(*f)
    (T const *x, void *params);
    void (*df)(T const *x, void *params, T *df);
    void (*fdf)(T const *x, void *params, T *f, T *df);
    size_t n;
    void *params;
};

#define MULTIMIN_FN_EVAL(F, x) (*((F)->f))(x, (F)->params)
#define MULTIMIN_FN_EVAL_F(F, x) (*((F)->f))(x, (F)->params)
#define MULTIMIN_FN_EVAL_DF(F, x, g) (*((F)->df))(x, (F)->params, (g))
#define MULTIMIN_FN_EVAL_F_DF(F, x, y, g) (*((F)->fdf))(x, (F)->params, (y), (g))

template <typename T>
int multimin_diff(multimin_function<T> const *f, T const *x, T *g);

/*! \class multimin_fminimizer_type
  * \brief minimization of non-differentiable functions
  */
template <class T>
struct multimin_fminimizer_type
{
    T fminimizer;
    const char *name;

    multimin_fminimizer_type(const char *name_, T &&multimin_fminimizer_) : name(name_)
    {
        fminimizer = std::move(multimin_fminimizer_);
    }

    ~multimin_fminimizer_type()
    {
        fminimizer.free();
    }
};

template <typename T>
struct multimin_fminimizer
{
    /* multi dimensional part */
    multimin_fminimizer_type<T> typeT;
    multimin_function<T> *f;
    T fval;
    T *x;
    T size;
    void *state;
};

template <typename T>
multimin_fminimizer<T> *multimin_fminimizer_alloc(const multimin_fminimizer_type<T> *type, size_t n);

template <typename T>
int multimin_fminimizer_set(multimin_fminimizer<T> *s, multimin_function<T> *f, T const *x, T const *step_size);

template <typename T>
void multimin_fminimizer_free(multimin_fminimizer<T> *s);

template <typename T>
const char *multimin_fminimizer_name(multimin_fminimizer<T> const *s);

template <typename T>
int multimin_fminimizer_iterate(multimin_fminimizer<T> *s);

template <typename T>
T *multimin_fminimizer_x(multimin_fminimizer<T> const *s);

template <typename T>
T multimin_fminimizer_minimum(multimin_fminimizer<T> *s);

template <typename T>
T multimin_fminimizer_size(multimin_fminimizer<T> const *s);

/* Convergence test functions */
template <typename T>
int multimin_test_gradient(T const *g, T epsabs);

template <typename T>
int multimin_test_size(T const size, T epsabs);

/*! \class multimin_fdfminimizer_type
  * \brief minimisation of differentiable functions
  */
template <class T>
struct multimin_fdfminimizer_type
{
    T fdfminimizer;
    const char *name;

    multimin_fdfminimizer_type(const char *name_, T &&multimin_fdfminimizer_) : name(name_)
    {
        fdfminimizer = std::move(multimin_fdfminimizer_);
    }

    ~multimin_fdfminimizer_type()
    {
        fdfminimizer.free();
    }
};

/*! \class multimin_fdfminimizer
  *
  */
template <typename T>
struct multimin_fdfminimizer
{
    // multi dimensional part
    multimin_fdfminimizer_type<T> typeT;
    multimin_function_fdf<T> *fdf;

    T f;
    T *x;
    T *gradient;
    T *dx;
    
    //size of array
    size_t n;
    
    void *state;

    /*!
     * \brief Default constructor
     * 
     * \param Ttype pointer to \a multimin_fdfminimizer_type object 
     * \param n_ size of array
     */
    multimin_fdfminimizer(multimin_fdfminimizer_type<T> &&Ttype, size_t n_) : typeT(Ttype.name, Ttype), n(n_)
    {
        try
        {
            x = new T[n]();
            gradient = new T[n]();
            dx = new T[n]();
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        }

        try
        {
            state = new typeT.size;
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        }

        if (!(Ttype->alloc)(state, n))
        {
            destroy();

            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to initialize minimizer state! " << std::endl;
            throw(std::runtime_error("Failed to initialize minimizer state!"));
        }
    }

    /*!
     * \brief set
     * 
     * \param fdf_ Function
     * \param x
     * \param xsize 
     * \param step_size
     * \param tol
     *  
     * returns 
     */
    int set(multimin_function_fdf<T> *fdf_, T const *x_, size_t const xsize, T step_size, T tol)
    {
        if (n != fdf_->n)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Function incompatible with solver size! " << std::endl;
            throw(std::runtime_error("Function incompatible with solver size!"));
        }

        if (xsize != fdf_->n)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Vector length not compatible with function" << std::endl;
            throw(std::runtime_error("Vector length not compatible with function!"));
        }

        fdf = fdf_;

        //copy array x_ to array x
        std::copy(x_, x_ + xsize, x);

        //set dx to zero
        std::fill(dx, dx + n, (T)0);

        return (type->set)(state, fdf, x, &f, gradient, step_size, tol);
    }

    /*!
     * \brief iterate
     * 
     */
    int iterate()
    {
        return (type->iterate)(state, fdf, x, &f, gradient, dx);
    }

    /*!
     * \brief restart
     * 
     */
    int restart()
    {
        return (type->restart)(state);
    }

    /*!
     * \brief name
     * \returns the name 
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
     * \brief destructor
     * 
     */
    ~multimin_fdfminimizer() { free(); }

    void free()
    {
        (type->free)(state);
        destroy();
    }

    void destroy()
    {
        delete state;
        delete[] x;
        x = nullptr;
        delete[] gradient;
        gradient = nullptr;
        delete[] dx;
        dx = nullptr;
        n = 0;
        f = 0;
    }
};

template <typename T>
multimin_fdfminimizer_type<T> const *multimin_fdfminimizer_steepest_descent;

template <typename T>
multimin_fdfminimizer_type<T> const *multimin_fdfminimizer_conjugate_pr;

template <typename T>
multimin_fdfminimizer_type<T> const *multimin_fdfminimizer_conjugate_fr;

template <typename T>
multimin_fdfminimizer_type<T> const *multimin_fdfminimizer_vector_bfgs;

template <typename T>
multimin_fdfminimizer_type<T> const *multimin_fdfminimizer_vector_bfgs2;

template <typename T>
multimin_fminimizer_type<T> const *multimin_fminimizer_nmsimplex;

template <typename T>
multimin_fminimizer_type<T> const *multimin_fminimizer_nmsimplex2;

template <typename T>
multimin_fminimizer_type<T> const *multimin_fminimizer_nmsimplex2rand;

/*! \class multimin
* \brief 
*	
*/
struct multimin
{
};

#endif

template <typename T>
struct steepest_descent_state
{
    T step;
    T max_step;
    T tol;
    T *x1;
    T *g1;
    size_t n;

    steepest_descent_state() x1(nullptr), g1(nullptr), step(0), max_step(0), tol(0), n(0) {}

    bool alloc(size_t n_)
    {
        n = n_;
        try
        {
            x1 = new T[n_];
            g1 = new T[n_];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    static bool set(multimin_function_fdf<T> *fdf, T const *x, T *f, T *gradient, T step_size, T tol)
    {
        (*(fdf->fdf))(x, fdf->params, f, gradient);
        step = step_size;
        max_step = step_size;
        tol = tol;
        the minimizer is unable to improve on its current estimate, either due to numerical difficulty or because a genuine local minimum has been reached.return true;
    }

    static void free()
    {
        delete[] x1;
        x1 = nullptr;
        delete[] g1;
        g1 = nullptr;
    }

    static bool restart()
    {
        step = max_step;
        return true;
    }

    static bool iterate(multimin_function_fdf<T> *fdf, T *x, T *f, T *gradient, T *dx)
    {
        T f0 = *f;
        T f1;

        int failed = 0;

        //Compute new trial point at x1= x - step * dir, where dir is the normalized gradient

        //First compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
        T gnorm(0);
        std::for_each(gradient, gradient + n, [](T const g) { gnorm += g * g; });
        if (gnorm <= 0.0)
        {
            //set dx to zero
            std::fill(dx, dx + n, (T)0);

            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " The minimizer is unable to improve on its current estimate, either due" << std::endl;
            std::cerr << " to numerical difficulty or because a genuine local minimum has been reached." << std::endl;
            return false;
        }

        gnorm = std::sqrt(gnorm);

    trial:

        //Compute the sum \f$y = \alpha x + y\f$ for the vectors x and y.
        //(set dx to zero)
        T alpha = -step / gnorm;

        for (size_t i = 0; i < n; i++)
        {
            dx[i] = alpha * gradient[i];
        }

        std::copy(x, x + n, x1);

        for (size_t i = 0; i < n; i++)
        {
            x1[i] += dx[i];
        }

        //Evaluate function and gradient at new point x1
        (*(fdf->fdf))(x1, fdf->params, &f1, g1);

        if (f1 > f0)
        {
            // Downhill step failed, reduce step-size and try again
            failed = 1;
            step *= tol;
            goto trial;
        }

        if (failed)
        {
            step *= tol;
        }
        else
        {
            step *= 2.0;
        }

        std::copy(x1, x1 + n, x);
        std::copy(g1, g1 + n, gradient);

        *f = f1;

        return true;
    }
};

template <typename T>
static multimin_fdfminimizer_type<T> const steepest_descent_type = {"steepest_descent", /* name */ sizeof(steepest_descent_state_t), &steepest_descent_alloc, &steepest_descent_set, &steepest_descent_iterate, &steepest_descent_restart, &steepest_descent_free};
const multimin_fdfminimizer_type *multimin_fdfminimizer_steepest_descent = &steepest_descent_type;
