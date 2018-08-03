#ifndef UMUQ_UNIFORMDISTRIBUTION_H
#define UMUQ_UNIFORMDISTRIBUTION_H

#include "../function/densityfunction.hpp"

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
template <typename T>
class uniformDistribution : public densityFunction<T, FUN_x<T>>
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
     * \brief Destroy the uniform Distribution object
     * 
     */
    ~uniformDistribution() {}

    /*!
     * \brief Uniform distribution density function
     * 
     * \param x  Input value
     * 
     * \returns  Density function value 
     */
    inline T uniformDistribution_f(T const x);

    /*!
     * \brief Log of Uniform distribution density function
     * 
     * \param x  Input value
     * 
     * \returns  Log of density function value
     */
    inline T uniformDistribution_lf(T const x);
};

/*!
 * \brief Construct a new exponential distribution object
 * 
 * \param mu Mean, \f$ \mu \f$
 */
template <typename T>
uniformDistribution<T>::uniformDistribution(T const mu) : densityFunction<T, FUN_x<T>>(&mu, 1, "exponential")
{
    this->f = std::bind(&uniformDistribution<T>::exponentialDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<T>::exponentialDistribution_lf, this, std::placeholders::_1);
}

/*!
 * \brief Exponential distribution density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T>
inline T uniformDistribution<T>::exponentialDistribution_f(T const x)
{
    return x < T{} ? T{} : std::exp(-x / this->params[0]) / this->params[0];
}

/*!
 * \brief Log of exponential distribution density function
 * 
 * \param x Input value
 * 
 * \returns  Log of density function value 
 */
template <typename T>
inline T uniformDistribution<T>::exponentialDistribution_lf(T const x)
{
    return x < this->params[0] ? std::numeric_limits<T>::infinity() : -std::log(this->params[0] - x / this->params[0]);
}

#endif //UMUQ_UNIFORMDISTRIBUTION_H