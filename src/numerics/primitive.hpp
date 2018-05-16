#ifndef UMUQ_PRIMITIVE_H
#define UMUQ_PRIMITIVE_H

/*! \class primitive
  * \brief A class for primitive function
  * 
  * \tparam T  data type
  * \tparam TF function type
  */
template <typename T, class TF>
class primitive
{
  public:
  /*!
   * /brief Primitive function
   * 
   * /param x input
   * /return value of \f$ {\mathbf C}(s) \f$
   */
    inline T f(T const *x)
    {
        return static_cast<TF *>(this)->f(x);
    }

  private:
    friend TF;
};

/*! \class quartic_spline
 * \brief Primitive function (quartic spline)
 * 
 * \tparam T data type
 * 
 * quartic spline function
 * \f$ 1 - 6 x^2 + 8 x^3 - 3 x^4 \f$
 * 
 * Reference: Chen et al., Int. J. Numer. Meth. Engng 2003; 56:935–960.
 */
template <typename T>
class quartic_spline : public primitive<T, quartic_spline<T>>
{
  public:
    /*! 
     * \brief Primitive function
     * 
     * \param  x  input 
     * \returns value of \f$ {\mathbf C}(s) = 1 - 6 x^2 + 8 x^3 - 3 x^4 \f$
     */
    inline T f(T const *x)
    {
        return (*x > static_cast<T>(1)) ? T{} : 1 + (*x) * (*x) * (-6 + (*x) * (8 - 3 * (*x)));
    }
};

/*! \class cubic_spline
 * \brief Primitive function (cubic spline)
 * 
 * \tparam T data type
 * 
 * cubic spline function
 * \f[ 
 * {\mathbf C}(s)=\left\{\begin{matrix} 
 * 6s^3 -6 s^2 + 1   &\text{for } &s\le \left(\frac{1}{2}\right) \\ 
 * -2 s^3+6 s^2-6s+2 &\text{for } &\left(\frac{1}{2}\right) < s \le 1 \\
 * 0                 &\text{for } &s > 1 \end{matrix} 
 * \right. 
 * \f]
 * 
 * Reference: Chen et al., Int. J. Numer. Meth. Engng 2003; 56:935–960.
 */
template <typename T>
class cubic_spline : public primitive<T, cubic_spline<T>>
{
  public:
    /*! 
     * \brief Primitive function
     * 
     * \param  x  input 
     * \returns value of \f$ {\mathbf C}(s) \f$
     */
    inline T f(T const *x)
    {
        return (*x > static_cast<T>(1)) ? T{} : (*x > static_cast<T>(0.5)) ? 2 - (*x) * (6 - (*x) * (6 - 2 * (*x))) : 1 - (*x) * (*x) * (6 - 6 * (*x));
    }
};

/*! \class normalizedgaussian
 * \brief Primitive function (normalized Gaussian)
 * 
 * \tparam T data type
 * 
 * normalized Gaussian function
 * \f[ 
 * {\mathbf C}(s)=\left\{\begin{matrix} 
 * \frac{e^{-\left(s/\alpha\right)^2} - e^{-\left(1/\alpha\right)^2}}{1-e^{-\left(1/\alpha\right)^2}} &\text{for } &s \le 1\\
 * 0                                                                                                  &\text{for } &s > 1
 * \right. 
 * \f]
 * 
 * Reference: Chen et al., Int. J. Numer. Meth. Engng 2003; 56:935–960.
 */
template <typename T>
class normalizedgaussian : public primitive<T, normalizedgaussian<T>>
{
  public:
    /*! 
     * \brief Primitive function
     * 
     * \param  x  input 
     * \returns value of \f$ {\mathbf C}(s) \f$
     */
    inline T f(T const *x)
    {
        return (*x > static_cast<T>(1)) ? T{} : (std::exp(-std::pow(*x * alpha, 2)) - std::exp(-alpha * alpha)) / (1 - std::exp(-alpha * alpha));
    }

  private:
    //! The \f$ \alpha \f$ parameter is taken to be 0.3, as commonly used in the literature.
    static T alpha = static_cast<T>(1) / static_cast<T>(0.3);
};

#endif
