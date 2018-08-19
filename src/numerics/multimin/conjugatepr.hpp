#ifndef UMUQ_CONJUGATEPR_H
#define UMUQ_CONJUGATEPR_H

/*! \class conjugatePr
 *  \ingroup multimin_Module
 * 
 * \brief Conjugate Polak-Ribiere gradient algorithm
 * It differs from the Fletcher-Reeves method, only in the choice of the coefficient \f$ \beta \f$.
 * It works well when the evaluation point is close enough to the minimum of the objective function 
 * that it is well approximated by a quadratic hypersurface. 
 * 
 * \tparam T Data type
 */
template <typename T>
class conjugatePr : public differentiableFunctionMinimizer<T>
{
  public:
    /*!
     * \brief Construct a new conjugate Pr object
     * 
     * \param Name Minimizer name
     */
    explicit conjugatePr(const char *Name = "conjugatePr");

    /*!
     * \brief Destroy the conjugate Pr object
     * 
     */
    ~conjugatePr();

    /*!
     * \brief Resizes the minimizer vectors to contain nDim elements
     *
     * \param nDim  New size of the minimizer vectors
     *
     * \returns true
     */
    bool reset(int const nDim) noexcept;

    /*!
     * \brief Initilize the minimizer
     * 
     * \return true 
     * \return false 
     */
    bool init();

    /*!
     * \brief Drives the iteration of each algorithm
     *
     * It performs one iteration to update the state of the minimizer.
     *
     * \return true
     * \return false If the iteration encounters an unexpected problem
     */
    bool iterate();

    /*!
     * \brief Restart the iterator
     * 
     * \return true 
     * \return false 
     */
    inline bool restart();

  private:
    //! Iteration
    int iter;

    //!
    std::vector<T> p;
    //!
    std::vector<T> x1;
    //!
    std::vector<T> dx1;
    //!
    std::vector<T> x2;
    //!
    std::vector<T> g0;

    //!
    T pnorm;
    //!
    T g0norm;
};

template <typename T>
conjugatePr<T>::conjugatePr(const char *Name) : differentiableFunctionMinimizer<T>(Name) {}

template <typename T>
conjugatePr<T>::~conjugatePr() {}

template <typename T>
bool conjugatePr<T>::reset(int const nDim) noexcept
{
    if (nDim <= 0)
    {
        UMUQFAILRETURN("Invalid number of parameters specified!");
    }

    this->x.resize(nDim);
    this->dx.resize(nDim);
    this->gradient.resize(nDim);

    p.resize(nDim);
    g0.resize(nDim);

    x1.resize(nDim, 0);
    dx1.resize(nDim, 0);
    x2.resize(nDim, 0);

    return true;
}

template <typename T>
bool conjugatePr<T>::init()
{
    iter = 0;

    // Use the gradient as the initial direction
    std::copy(this->gradient.begin(), this->gradient.end(), p.begin());
    std::copy(this->gradient.begin(), this->gradient.end(), g0.begin());

    {
        // First compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
        T s(0);
        std::for_each(this->gradient.begin(), this->gradient.end(), [&](T const g_i) { s += g_i * g_i; });

        pnorm = std::sqrt(s);
        g0norm = pnorm;
    }

    return true;
}

template <typename T>
bool conjugatePr<T>::iterate()
{
    int const n = this->getDimension();

    T fa = this->fval;
    T fb;
    T dir;
    T stepa(0);
    T stepb;
    T stepc = this->step;
    T g1norm;

    if (pnorm <= T{} || g0norm <= T{})
    {
        // Set dx to zero
        std::fill(this->dx.begin(), this->dx.end(), T{});

        UMUQFAILRETURN("The minimizer is unable to improve on its current estimate, either due \n to the numerical difficulty or because a genuine local minimum has been reached.");
    }

    T pg(0);

    // Determine which direction is downhill, +p or -p
    for (int i = 0; i < n; i++)
    {
        pg += p[i] * this->gradient[i];
    }

    dir = (pg >= T{}) ? static_cast<T>(1) : -static_cast<T>(1);

    // Compute new trial point at x_c= x - step * p, where p is the current direction
    this->takeStep(this->x, p, stepc, dir / pnorm, x1, this->dx);

    // Evaluate function and gradient at new point xc
    T fc = this->fun.f(x1.data());

    if (fc < fa)
    {
        // Success, reduced the function value
        this->step = stepc * 2;

        this->fval = fc;

        std::copy(x1.begin(), x1.end(), this->x.begin());

        return this->fun.df(x1.data(), this->gradient.data());
    }

#ifdef DEBUG
    std::cout << "Got stepc = " << stepc << "fc = " << fc << std::endl;
#endif

    // Do a line minimisation in the region (xa,fa) (xc,fc) to find an
    // intermediate (xb,fb) satisifying fa > fb < fc.  Choose an initial
    // xb based on parabolic interpolation
    this->intermediatePoint(this->x, p, dir / pnorm, pg, stepc, fa, fc, x1, dx1, this->gradient, stepb, fb);

    if (stepb == T{})
    {
        UMUQFAILRETURN("The minimizer is unable to improve on its current estimate, either due \n to the numerical difficulty or because a genuine local minimum has been reached.");
    }

    this->minimize(this->x, p, dir / pnorm, stepa, stepb, stepc, fa, fb, fc, this->tol, x1, dx1, x2, this->dx, this->gradient, this->step, this->fval, g1norm);

    std::copy(x2.begin(), x2.end(), this->x.begin());

    // Choose a new conjugate direction for the next step
    iter = (iter + 1) % n;

    if (iter == 0)
    {
        std::copy(this->gradient.begin(), this->gradient.end(), p.begin());

        pnorm = g1norm;
    }
    else
    {
        // \f$ p' = g1 - beta * p \f$

        // \f$ g0' = g0 - g1 \f$
        for (int i = 0; i < n; i++)
        {
            g0[i] -= this->gradient[i];
        }

        T g0g1(0);
        // g1g0 = (g0-g1).g1
        for (int i = 0; i < n; i++)
        {
            g0g1 += g0[i] * this->gradient[i];
        }

        // \f$ \beta = -((g1 - g0).g1)/(g0.g0) \f$
        T const beta = -g0g1 / (g0norm * g0norm);

        std::for_each(p.begin(), p.end(), [&](T &p_i) { p_i *= beta; });

        for (int i = 0; i < n; i++)
        {
            p[i] += this->gradient[i];
        }

        T s(0);
        std::for_each(p.begin(), p.end(), [&](T const p_i) { s += p_i * p_i; });
        pnorm = std::sqrt(s);
    }

    g0norm = g1norm;

    std::copy(this->gradient.begin(), this->gradient.end(), g0.begin());

#ifdef DEBUG
    std::cout << "updated conjugate directions" << std::endl;
    //TODO print vector p and g
#endif

    return true;
}

template <typename T>
inline bool conjugatePr<T>::restart()
{
    iter = 0;
    return true;
}

#endif
