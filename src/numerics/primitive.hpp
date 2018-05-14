#ifndef UMUQ_PRIMITIVE_H
#define UMUQ_PRIMITIVE_H

/*! \class primitive
  * \brief Primitive function
  * 
  * \tparam T   data type
  * \tparan TF function type
  */
template <typename T, class TF>
class primitive
{
  public:
    inline T f(T const *x)
    {
        return static_cast<TF *>(this)->f(x);
    }

  private:
    friend TF;
};

/*!
 * \brief Primitive function (quartic spline)
 * 
 * Reference: Chen et al., Int. J. Numer. Meth. Engng 2003; 56:935–960.
 * 
 * \returns \f$ 1 - 6 x^2 + 8 x^3 - 3 x^4 \f$
 */
template <typename T>
class quartic_spline : public primitive<T, quartic_spline<T>>
{
  public:
    inline T f(T const *x)
    {
        return (*x > static_cast<T>(1)) ? T{} : 1 + (*x) * (*x) * (-6 + (*x) * (8 - 3 * (*x)));
    }
};

/*!
 * \brief Primitive function (cubic spline)
 * 
 * Reference: Chen et al., Int. J. Numer. Meth. Engng 2003; 56:935–960.
 * 
 * \returns \f$ {\mathbf C}(s) \f$
 * 
 * \f$ {\mathbf C}(s)=\left\{\begin{matrix} 6s^3 -6 s^2 + 1  & \text{for } &s\le \left(\frac{1}{2}\right)\\ -2 s^3+6 s^2-6s+2 & \text{for } & \left(\frac{1}{2}\right) < s \le 1 \\0  & \text{for } &s > 1 \end{matrix} \right. \f$
 * 
 */
template <typename T>
class cubic_spline : public primitive<T, cubic_spline<T>>
{
  public:
    inline T f(T const *x)
    {
        return (*x > static_cast<T>(1)) ? T{} : (*x > static_cast<T>(0.5)) ? 2 - (*x) * (6 - (*x) * (6 - 2 * (*x))) : 1 - (*x) * (*x) * (6 - 6 * (*x));
    }
};

/*!
 * \brief Primitive function (normalized Gaussian)
 * 
 * Reference: Chen et al., Int. J. Numer. Meth. Engng 2003; 56:935–960.
 * 
 * \returns \f$ {\mathbf C}(s) \f$
 * 
 * \f$ {\mathbf C}(s)=\left\{\begin{matrix} 6s^3 -6 s^2 + 1  & \text{for } &s\le \left(\frac{1}{2}\right)\\ -2 s^3+6 s^2-6s+2 & \text{for } & \left(\frac{1}{2}\right) < s \le 1 \\0  & \text{for } &s > 1 \end{matrix} \right. \f$
 * 
 */
template <typename T>
class normalizedgaussian : public primitive<T, normalizedgaussian<T>>
{
  public:
    inline T f(T const *x)
    {
        return (*x > static_cast<T>(1)) ? T{} : (std::exp(-std::pow(*x * alpha, 2)) - std::exp(-alpha * alpha)) / (1 - std::exp(-alpha * alpha));
    }

  private:
    static T alpha = static_cast<T>(1) / static_cast<T>(0.3);
};

#endif
