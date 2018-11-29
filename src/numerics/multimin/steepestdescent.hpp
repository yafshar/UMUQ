#ifndef UMUQ_STEEPESTDESCENT_H
#define UMUQ_STEEPESTDESCENT_H

namespace umuq
{

inline namespace multimin
{

/*! \class steepestDescent
 * \ingroup Multimin_Module
 * 
 * \brief steepestDescent for differentiable function minimizer type
 * 
 * The steepest descent algorithm follows the downhill gradient of the function at each step. 
 * When a downhill step is successful the step-size is increased by a factor of two. 
 * If the downhill step leads to a higher function value then the algorithm backtracks 
 * and the step size is decreased using the parameter tol. 
 * 
 * A suitable value of tol for most applications is 0.1. 
 * The steepest descent method is inefficient and is included only for demonstration purposes. 
 * 
 * \tparam DataType Data type
 */
template <typename DataType>
class steepestDescent : public differentiableFunctionMinimizer<DataType>
{
  public:
    /*!
     * \brief Construct a new steepest Descent object
     * 
     * \param Name Minimizer name
     */
    explicit steepestDescent(char const *Name = "steepestDescent");

    /*!
     * \brief Destroy the steepest Descent object
     * 
     */
    ~steepestDescent();

    /*!
     * \brief Resizes the minimizer vectors to contain nDim elements
     *
     * \param nDim  New size of the minimizer vectors
     *
     * \returns true
     */
    bool reset(int const nDim) noexcept;

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
    //!
    std::vector<DataType> x1;
    //!
    std::vector<DataType> g1;
};

template <typename DataType>
steepestDescent<DataType>::steepestDescent(const char *Name) : differentiableFunctionMinimizer<DataType>(Name) {}

template <typename DataType>
steepestDescent<DataType>::~steepestDescent() {}

template <typename DataType>
bool steepestDescent<DataType>::reset(int const nDim) noexcept
{
    if (nDim <= 0)
    {
        UMUQFAILRETURN("Invalid number of parameters specified!");
    }

    this->x.resize(nDim);
    this->dx.resize(nDim);
    this->gradient.resize(nDim);

    x1.resize(nDim, 0);
    g1.resize(nDim, 0);

    return true;
}

template <typename DataType>
bool steepestDescent<DataType>::iterate()
{
    int const n = this->getDimension();

    DataType f0 = this->fval;

    bool failed(false);

    // Compute new trial point at x1= x - step * dir, where dir is the normalized gradient

    // First compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
    DataType gnorm(0);
    std::for_each(this->gradient.begin(), this->gradient.end(), [&](DataType const g_i) { gnorm += g_i * g_i; });

    if (gnorm <= DataType{})
    {
        // set dx to zero
        std::fill(this->dx.begin(), this->dx.end(), DataType{});

        UMUQFAILRETURN("The minimizer is unable to improve on its current estimate, either due \n to the numerical difficulty or because a genuine local minimum has been reached!");
    }

    gnorm = std::sqrt(gnorm);

    DataType f1 = 2 * f0;
    while (f1 > f0)
    {
        // Compute the sum \f$y = \alpha x + y\f$ for the vectors x and y.
        // (set dx to zero)
        DataType const alpha = -this->step / gnorm;
        for (int i = 0; i < n; i++)
        {
            this->dx[i] = alpha * this->gradient[i];
        }

        std::copy(this->x.begin(), this->x.end(), x1.begin());

        for (int i = 0; i < n; i++)
        {
            x1[i] += this->dx[i];
        }

        // Evaluate function and gradient at new point x1
        this->fun.fdf(x1.data(), &f1, g1.data());

        if (f1 > f0)
        {
            // Downhill step failed, reduce step-size and try again
            failed = true;

            this->step *= this->tol;
        }
    }

    failed ? this->step *= this->tol : this->step *= 2;

    std::copy(x1.begin(), x1.end(), this->x.begin());
    std::copy(g1.begin(), g1.end(), this->gradient.begin());

    this->fval = f1;

    return true;
}

template <typename DataType>
inline bool steepestDescent<DataType>::restart()
{
    this->step = this->maxStep;
    return true;
}

} // namespace multimin
} // namespace umuq

#endif // UMUQ_STEEPESTDESCENT
