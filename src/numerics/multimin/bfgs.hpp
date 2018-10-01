#ifndef UMUQ_BFGS_H
#define UMUQ_BFGS_H

namespace umuq
{
/*! \namespace multimin
 * \brief Namespace containing all the functions for Multidimensional Minimization Module
 * 
 * It includes all the functionalities for finding minima of arbitrary multidimensional 
 * functions. It provides low level components for a variety of iterative minimizers 
 * and convergence tests.
 */
inline namespace multimin
{

/*! \class bfgs
 *  \ingroup multimin_Module
 * 
 * \brief Limited memory Broyden-Fletcher-Goldfarb-Shanno method
 * 
 * This is a quasi-Newton method which builds up an approximation to the second derivatives of the 
 * function f using the difference between successive gradient vectors. 
 * By combining the first and second derivatives the algorithm is able to take Newton-type steps 
 * towards the function minimum, assuming quadratic behavior in that region.
 * 
 * \tparam T  Data type
 */
template <typename T>
class bfgs : public differentiableFunctionMinimizer<T>
{
  public:
    /*!
     * \brief Construct a new bfgs object
     * 
     * \param Name Minimizer name
     */
    explicit bfgs(char const *Name = "bfgs");

    /*!
     * \brief Destroy the bfgs object
     * 
     */
    ~bfgs();

    /*!
     * \brief Resizes the minimizer vectors to contain nDim elements
     *
     * \param nDim  New size of the minimizer vectors
     *
     * \returns true
     */
    bool reset(int const nDim) noexcept;

    /*!
     * \brief Initialize the minimizer
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
    std::vector<T> x0;
    //!
    std::vector<T> x1;
    //!
    std::vector<T> x2;
    //!
    std::vector<T> g0;
    //!
    std::vector<T> dx0;
    //!
    std::vector<T> dx1;
    //!
    std::vector<T> dg0;

    //!
    T pnorm;
    //!
    T g0norm;
};

template <typename T>
bfgs<T>::bfgs(const char *Name) : differentiableFunctionMinimizer<T>(Name) {}

template <typename T>
bfgs<T>::~bfgs() {}

template <typename T>
bool bfgs<T>::reset(int const nDim) noexcept
{
    if (nDim <= 0)
    {
        UMUQFAILRETURN("Invalid number of parameters specified!");
    }

    this->x.resize(nDim);
    this->dx.resize(nDim);
    this->gradient.resize(nDim);

    p.resize(nDim, 0);
    x0.resize(nDim, 0);
    x1.resize(nDim, 0);
    x2.resize(nDim, 0);
    g0.resize(nDim, 0);
    dx0.resize(nDim, 0);
    dx1.resize(nDim, 0);
    dg0.resize(nDim, 0);

    return true;
}

template <typename T>
bool bfgs<T>::init()
{
    iter = 0;

    // Use the gradient as the initial direction
    std::copy(this->x.begin(), this->x.end(), x0.begin());
    std::copy(this->gradient.begin(), this->gradient.end(), p.begin());
    std::copy(this->gradient.begin(), this->gradient.end(), g0.begin());

    {
        // Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
        T s(0);
        std::for_each(this->gradient.begin(), this->gradient.end(), [&](T const g_i) { s += g_i * g_i; });

        pnorm = std::sqrt(s);
        g0norm = pnorm;
    }

    return true;
}

template <typename T>
bool bfgs<T>::iterate()
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
    std::cout << "Got stepc = " << stepc << " fc = " << fc << std::endl;
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
        // This is the BFGS update:
        // \f$ p' = g1 - A dx - B dg \f$
        // \f$ A = - (1+ dg.dg/dx.dg) B + dg.g/dx.dg \f$
        // \f$ B = dx.g/dx.dg \f$
        T dxg;
        T dgg;
        T dxdg;
        T dgnorm;
        T A;
        T B;

        // \f$ dx0 = x - x0 \f$
        std::copy(this->x.begin(), this->x.end(), dx0.begin());

        for (int i = 0; i < n; i++)
        {
            dx0[i] -= x0[i];
        }

        // \f$ dg0 = g - g0 \f$
        std::copy(this->gradient.begin(), this->gradient.end(), dg0.begin());

        for (int i = 0; i < n; i++)
        {
            dg0[i] -= g0[i];
        }

        dxg = T{};
        for (int i = 0; i < n; i++)
        {
            dxg += dx0[i] * this->gradient[i];
        }

        dgg = T{};
        for (int i = 0; i < n; i++)
        {
            dgg += dg0[i] * this->gradient[i];
        }

        dxdg = T{};
        for (int i = 0; i < n; i++)
        {
            dxdg += dx0[i] * dg0[i];
        }

        {
            T s(0);
            std::for_each(dg0.begin(), dg0.end(), [&](T const d_i) { s += d_i * d_i; });

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

        std::copy(this->gradient.begin(), this->gradient.end(), p.begin());

        for (int i = 0; i < n; i++)
        {
            p[i] -= A * dx0[i];
        }

        for (int i = 0; i < n; i++)
        {
            p[i] -= B * dg0[i];
        }

        {
            T s(0);
            std::for_each(p.begin(), p.end(), [&](T const p_i) { s += p_i * p_i; });

            pnorm = std::sqrt(s);
        }
    }

    std::copy(this->gradient.begin(), this->gradient.end(), g0.begin());

    std::copy(this->x.begin(), this->x.end(), x0.begin());

    {
        T s(0);
        std::for_each(g0.begin(), g0.end(), [&](T const g_i) { s += g_i * g_i; });

        g0norm = std::sqrt(s);
    }

#ifdef DEBUG
    std::cout << "updated directions" << std::endl;
    //TODO print vector p and g
#endif

    return true;
}

template <typename T>
inline bool bfgs<T>::restart()
{
    iter = 0;
    return true;
}

} // namespace multimin
} // namespace umuq

#endif // UMUQ_BFGS
