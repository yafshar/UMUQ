#ifndef UMUQ_UNIFORMDISTRIBUTION_H
#define UMUQ_UNIFORMDISTRIBUTION_H

namespace umuq
{
/*! \namespace density
 * \brief Namespace containing all the functions for probability density computation
 *
 */
inline namespace density
{

/*! \class uniformDistribution
 * \brief Flat (Uniform) distribution function
 * 
 * This class provides probability density \f$ p(x) \f$ and it's Log at x for a uniform distribution 
 * from \f$ \[a \cdots b\] \f$, 
 * using: 
 * \f[
 * p(x)= \left\{\begin{matrix}
 * 1/(b-a)  &a \leqslant  x < b \\ 
 *  0       &otherwise
 * \end{matrix}\right.
 * \f]
 * 
 * \tparam T Data type
 */
template <typename T, class V = T const *>
class uniformDistribution : public densityFunction<T, std::function<T(V)>>
{
  public:
    /*!
     * \brief Construct a new uniform Distribution object
     * 
     * \param a  Lower bound
     * \param b  Upper bound
     */
    uniformDistribution(T const a, T const b);

    /*!
     * \brief Construct a new uniform Distribution object
     * 
     * \param a  Lower bound
     * \param b  Upper bound
     * \param n  Total number of Lower bound + Upper bound inputs
     */
    uniformDistribution(T const *a, T const *b, int const n);

    /*!
     * \brief Destroy the uniform Distribution object
     * 
     */
    ~uniformDistribution();

    /*!
     * \brief Uniform distribution density function
     * 
     * \param x  Input value
     * 
     * \returns  Density function value 
     */
    inline T uniformDistribution_f(T const *x);

    /*!
     * \brief Log of Uniform distribution density function
     * 
     * \param x  Input value
     * 
     * \returns  Log of density function value
     */
    inline T uniformDistribution_lf(T const *x);

  private:
    //! Const value for uniform distribution function
    T uniformDistribution_fValue;
    //! Const value for logarithm of the uniform distribution function
    T uniformDistribution_lfValue;

    //! Helper function
    inline T uniformDistribution_f_();
    //! Helper log function
    inline T uniformDistribution_lf_();
};

template <typename T, class V>
uniformDistribution<T, V>::uniformDistribution(T const a, T const b) : densityFunction<T, std::function<T(V)>>(&a, &b, 2, "uniform")
{
    this->f = std::bind(&uniformDistribution<T>::uniformDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<T>::uniformDistribution_lf, this, std::placeholders::_1);
    uniformDistribution_fValue = uniformDistribution_f_();
    uniformDistribution_lfValue = uniformDistribution_lf_();
}

template <typename T, class V>
uniformDistribution<T, V>::uniformDistribution(T const *a, T const *b, int const n) : densityFunction<T, std::function<T(V)>>(a, b, n, "uniform")
{
    if (n % 2 != 0)
    {
        UMUQFAIL("Wrong number of inputs!");
    }
    this->f = std::bind(&uniformDistribution<T>::uniformDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<T>::uniformDistribution_lf, this, std::placeholders::_1);
    uniformDistribution_fValue = uniformDistribution_f_();
    uniformDistribution_lfValue = uniformDistribution_lf_();
}

template <typename T, class V>
uniformDistribution<T, V>::~uniformDistribution() {}

template <typename T, class V>
inline T uniformDistribution<T, V>::uniformDistribution_f(T const *x)
{
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        if (x[i] < this->params[k] || x[i] >= this->params[k + 1])
        {
            return T{};
        }
    }
    return uniformDistribution_fValue;
}

template <typename T, class V>
inline T uniformDistribution<T, V>::uniformDistribution_lf(T const *x)
{
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        if (x[i] < this->params[k] || x[i] >= this->params[k + 1])
        {
            return std::numeric_limits<T>::infinity();
        }
    }
    return uniformDistribution_lfValue;
}

template <typename T, class V>
inline T uniformDistribution<T, V>::uniformDistribution_f_()
{
    T sum(1);
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        sum *= static_cast<T>(1) / (this->params[k + 1] - this->params[k]);
    }
    return sum;
}

template <typename T, class V>
inline T uniformDistribution<T, V>::uniformDistribution_lf_()
{
    T sum(0);
    for (std::size_t i = 0, k = 0; i < this->numParams / 2; i++, k += 2)
    {
        sum -= std::log(this->params[k + 1] - this->params[k]);
    }
    return sum;
}

} // namespace density
} // namespace umuq

#endif //UMUQ_UNIFORMDISTRIBUTION_H