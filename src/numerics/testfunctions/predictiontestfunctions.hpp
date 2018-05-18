#ifndef UMUQ_PREDICTIONTESTFUNCTIONS_H
#define UMUQ_PREDICTIONTESTFUNCTIONS_H

/*! \class franke2d
 * \brief Franke's Function class
 * 
 * \tparam T data type
 */
template <typename T>
class franke2d
{
  public:
    /*!
     * \brief Construct a new franke2d object
     * 
     */
    franke2d() {}

    /*! \fn f
     * \brief Franke's bivariate function
     * 
     * Franke's bivariate function is a weighted sum of four exponentials
     * \f[
     * f(x) &= 0.75 e^\left(-\frac{(9x_1-2)^2}{4} - \frac{(9x_2-2)^2}{4} \right) \\
     *      &+ 0.75 e^\left(-\frac{(9x_1+1)^2}{49} - \frac{(9x_2+1)}{10} \right) \\
     *      &+ 0.5 e^\left(-\frac{(9x_1-7)^2}{4} - \frac{(9x_2-3)^2}{4} \right) \\
     *      &- 0.2 e^\left(-(9x_1-4)^2 - (9x_2-7)^2 \right)
     * \f]
     * 
     * \param  x  input data point
     * 
     * \return f  function value at input data point
     */
    inline T f(T const *x)
    {
        T const x1 = x[0];
        T const x2 = x[1];
        T const t1 = 0.75 * std::exp(-std::pow(9 * x1 - 2, 2) / 4 - std::pow(9 * x2 - 2, 2) / 4);
        T const t2 = 0.75 * std::exp(-std::pow(9 * x1 + 1, 2) / 49 - (9 * x2 + 1) / 10);
        T const t3 = 0.5 * std::exp(-std::pow(9 * x1 - 7, 2) / 4 - std::pow(9 * x2 - 3, 2) / 4);
        T const t4 = -0.2 * std::exp(-std::pow(9 * x1 - 4, 2) - std::pow(9 * x2 - 7, 2));
        return t1 + t2 + t3 + t4;
    }
};


/*! \class rastrigin
 * \brief Rastrigin's Function class
 * 
 * \tparam T data type
 */
template <typename T>
class rastrigin
{
  public:
    /*!
     * \brief Construct a new Rastrigin's function object
     * 
     * \param dim Dimension of the space (default is 2)
     */
    rastrigin(int dim = 2) : nDim(dim) {}

    /*! \fn f
     * \brief Rastrigin function
     * 
     * The Rastrigin function has several local minima. 
     * It is highly multimodal, but locations of the minima are regularly distributed.
     * \f$ f(x) &= 10d + \sum_{i=1}^{d} \left[x_i^2 -10 cos \left( 2 \pi x_i\right) \right] \f$
     * 
     * \param  x  input data point
     * 
     * \return f  function value at input data point
     */
    inline T f(T const *x)
    {
        T sum(0);
        std::for_each(x, x + nDim, [&](T const i) { sum += i * i - 10 * std::cos(M_2PI * i); });
        return 10 * nDim + sum;
    }

  private:
    int nDim;
};

#endif
