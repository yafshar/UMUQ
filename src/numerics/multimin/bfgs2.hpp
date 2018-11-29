#ifndef UMUQ_BFGS2_H
#define UMUQ_BFGS2_H

#include "../function/utilityfunction.hpp"

namespace umuq
{

inline namespace multimin
{

/*! \class bfgs2
 * \ingroup Multimin_Module
 * 
 * \brief Limited memory Broyden-Fletcher-Goldfarb-Shanno method
 * 
 * \tparam DataType  Data type
 * 
 * Fletcher's implementation of the BFGS method, using the line minimization algorithm from 
 * 
 * The original BFGS is a quasi-Newton method which builds up an approximation to the second 
 * derivatives of the function f using the difference between successive gradient vectors. 
 * By combining the first and second derivatives the algorithm is able to take Newton-type steps 
 * towards the function minimum, assuming quadratic behavior in that region.
 * 
 * BFGS2 minimizer is the most efficient version available. 
 * It supersedes the bfgs algorithm and requires fewer function and gradient evaluations. 
 * The user-supplied tolerance tol corresponds to the parameter \f$ \sigma \f$ used by Fletcher. 
 * A value of 0.1 is recommended for typical use (larger values correspond to less accurate line searches).
 *
 * Reference:<br>
 * R. Fletcher, Practical Methods of Optimization (Second Edition) Wiley (1987), ISBN 0471915475.
 * Algorithms 2.6.2 and 2.6.4.
 *
 */
template <typename DataType>
class bfgs2 : public differentiableFunctionMinimizer<DataType>
{
  public:
    /*!
     * \brief Construct a new bfgs2 object
     * 
     * \param Name Minimizer name
     */
    explicit bfgs2(char const *Name = "bfgs2");

    /*!
     * \brief Destroy the bfgs2 object
     * 
     */
    ~bfgs2();

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

    //! f'(0) for f(x-alpha*p)
    DataType fp0;
    //!
    std::vector<DataType> p;
    //!
    std::vector<DataType> x0;
    //!
    std::vector<DataType> g0;
    //! Work space
    std::vector<DataType> dx0;
    //!
    std::vector<DataType> dg0;
    //!
    std::vector<DataType> x_alpha;
    //!
    std::vector<DataType> g_alpha;
    //!
    DataType pnorm;
    //!
    DataType g0norm;
    //!
    DataType delta_f;

    //! Wrapper function
    linearFunctionWrapper<DataType> funw;

    //! Minimization parameters
    DataType rho;
    DataType sigma;
    DataType tau1;
    DataType tau2;
    DataType tau3;

    int order;
};

template <typename DataType>
bfgs2<DataType>::bfgs2(const char *Name) : differentiableFunctionMinimizer<DataType>(Name) {}

template <typename DataType>
bfgs2<DataType>::~bfgs2() {}

template <typename DataType>
bool bfgs2<DataType>::reset(int const nDim) noexcept
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
    g0.resize(nDim, 0);
    dx0.resize(nDim, 0);
    dg0.resize(nDim, 0);
    x_alpha.resize(nDim, 0);
    g_alpha.resize(nDim, 0);

    return true;
}

template <typename DataType>
bool bfgs2<DataType>::init()
{
    iter = 0;

    delta_f = DataType{};

    // Use the gradient as the initial direction
    std::copy(this->x.begin(), this->x.end(), x0.begin());
    std::copy(this->gradient.begin(), this->gradient.end(), g0.begin());

    {
        // Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
        DataType s(0);
        std::for_each(g0.begin(), g0.end(), [&](DataType const g_i) { s += g_i * g_i; });

        g0norm = std::sqrt(s);
    }

    std::copy(this->gradient.begin(), this->gradient.end(), p.begin());

    {
        DataType const alpha = -static_cast<DataType>(1) / g0norm;
        std::for_each(p.begin(), p.end(), [&](DataType &p_i) { p_i *= alpha; });
    }

    {
        // Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
        DataType s(0);
        std::for_each(p.begin(), p.end(), [&](DataType const p_i) { s += p_i * p_i; });

        // should be 1
        pnorm = std::sqrt(s);
    }

    fp0 = -g0norm;

    // Set the Multidimensional function for the wrapper
    if (funw.set(this->fun))
    {
        int const n = this->getDimension();

        // Prepare the wrapper
        funw.prepare(n, x0.data(), this->fval, g0.data(), p.data(), x_alpha.data(), g_alpha.data());

        // Prepare 1d minimization parameters
        rho = static_cast<DataType>(0.01);
        sigma = this->tol;
        tau1 = static_cast<DataType>(9);
        tau2 = static_cast<DataType>(0.05);
        tau3 = static_cast<DataType>(0.5);

        // Use cubic interpolation where possible
        order = 3;

        return true;
    }

    return false;
}

template <typename DataType>
bool bfgs2<DataType>::iterate()
{
    if (pnorm == DataType{} || g0norm == DataType{} || fp0 == DataType{})
    {
        // set dx to zero
        std::fill(this->dx.begin(), this->dx.end(), DataType{});

        UMUQFAILRETURN("The minimizer is unable to improve on its current estimate, either due \n to the numerical difficulty or because a genuine local minimum has been reached!");
    }

    int const n = this->getDimension();

    DataType alpha(0);
    DataType alpha1;

    DataType pg;
    DataType dir;

    DataType f0 = this->fval;

    if (delta_f < DataType{})
    {
        DataType del = std::max(-delta_f, 10 * std::numeric_limits<DataType>::epsilon() * std::abs(f0));

        alpha1 = std::min(static_cast<DataType>(1), -2 * del / fp0);
    }
    else
    {
        alpha1 = std::abs(this->step);
    }

    // Line minimization, with cubic interpolation (order = 3)
    bool status = minimize<DataType>(funw, rho, sigma, tau1, tau2, tau3, order, alpha1, &alpha);

    if (status != true)
    {
        return false;
    }

    funw.updatePosition(alpha, this->x.data(), &this->fval, this->gradient.data());

    delta_f = this->fval - f0;

    // Choose a new direction for the next step
    {
        // This is the BFGS update:
        // \f$ p' = g1 - A dx - B dg \f$
        // \f$ A = - (1+ dg.dg/dx.dg) B + dg.g/dx.dg \f$
        // \f$ B = dx.g/dx.dg \f$

        //\f$ dx0 = x - x0 \f$
        std::copy(this->x.begin(), this->x.end(), dx0.begin());

        for (int i = 0; i < n; i++)
        {
            dx0[i] -= x0[i];
        }

        // keep a copy
        std::copy(dx0.begin(), dx0.end(), this->dx.begin());

        // \f$ dg0 = g - g0 \f$
        std::copy(this->gradient.begin(), this->gradient.end(), dg0.begin());

        for (int i = 0; i < n; i++)
        {
            dg0[i] -= g0[i];
        }

        DataType dxg(0);
        for (int i = 0; i < n; i++)
        {
            dxg += dx0[i] * this->gradient[i];
        }

        DataType dgg(0);
        for (int i = 0; i < n; i++)
        {
            dgg += dg0[i] * this->gradient[i];
        }

        DataType dxdg(0);
        for (int i = 0; i < n; i++)
        {
            dxdg += dx0[i] * dg0[i];
        }

        DataType dgnorm;
        {
            DataType s(0);
            std::for_each(dg0.begin(), dg0.end(), [&](DataType const d_i) { s += d_i * d_i; });

            dgnorm = std::sqrt(s);
        }

        DataType A;
        DataType B;

        if (dxdg != DataType{})
        {
            B = dxg / dxdg;
            A = -(static_cast<DataType>(1) + dgnorm * dgnorm / dxdg) * B + dgg / dxdg;
        }
        else
        {
            B = DataType{};
            A = DataType{};
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
    }

    std::copy(this->gradient.begin(), this->gradient.end(), g0.begin());

    std::copy(this->x.begin(), this->x.end(), x0.begin());

    {
        DataType s(0);
        std::for_each(g0.begin(), g0.end(), [&](DataType const g_i) { s += g_i * g_i; });

        g0norm = std::sqrt(s);
    }

    {
        DataType s(0);
        std::for_each(p.begin(), p.end(), [&](DataType const p_i) { s += p_i * p_i; });

        pnorm = std::sqrt(s);
    }

    //Update direction and fp0
    pg = DataType{};
    for (int i = 0; i < n; i++)
    {
        pg += p[i] * this->gradient[i];
    }

    dir = (pg >= DataType{}) ? -static_cast<DataType>(1) : static_cast<DataType>(1);

    {
        DataType const alpha = dir / pnorm;
        std::for_each(p.begin(), p.end(), [&](DataType &p_i) { p_i *= alpha; });
    }

    {
        DataType s(0);
        std::for_each(p.begin(), p.end(), [&](DataType const p_i) { s += p_i * p_i; });

        pnorm = std::sqrt(s);
    }

    fp0 = DataType{};
    for (int i = 0; i < n; i++)
    {
        fp0 += g0[i] * p[i];
    }

    funw.changeDirection();

    return true;
}

template <typename DataType>
inline bool bfgs2<DataType>::restart()
{
    iter = 0;
    return true;
}

} // namespace multimin
} // namespace umuq

#endif // UMUQ_BFGS2
