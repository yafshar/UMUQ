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
    ~uniformDistribution();

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

template <typename T>
uniformDistribution<T>::uniformDistribution(T const a, T const b) : densityFunction<T, FUN_x<T>>(std::vector<T>{a, b}.data(), 2, "uniform")
{
    this->f = std::bind(&uniformDistribution<T>::uniformDistribution_f, this, std::placeholders::_1);
    this->lf = std::bind(&uniformDistribution<T>::uniformDistribution_lf, this, std::placeholders::_1);
}

template <typename T>
uniformDistribution<T>::~uniformDistribution() {}

template <typename T>
inline T uniformDistribution<T>::uniformDistribution_f(T const x)
{
    return (x < this->params[1] && x >= this->params[0]) ? static_cast<T>(1) / (this->params[1] - this->params[0]) : T{};
}

template <typename T>
inline T uniformDistribution<T>::uniformDistribution_lf(T const x)
{
    return (x < this->params[1] && x >= this->params[0]) ? -std::log(this->params[1] - this->params[0]) : std::numeric_limits<T>::infinity();
}

#endif //UMUQ_UNIFORMDISTRIBUTION_H