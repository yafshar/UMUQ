#ifndef UMUQ_PRIMITIVE_H
#define UMUQ_PRIMITIVE_H

#include <cmath>

/*! \class primitive
 * \ingroup Numerics_Module
 *
 * \brief A class for primitive function
 *
 * \tparam DataType     Data type
 * \tparam FunctionType Function type
 */
template <typename DataType, class FunctionType>
class primitive
{
public:
  /*!
   * /brief Primitive function
   *
   * /param x input
   * /return value of \f$ {\mathbf C}(s) \f$
   */
  inline DataType f(DataType const *x)
  {
    return static_cast<FunctionType *>(this)->f(x);
  }

private:
  friend FunctionType;
};

/*! \class quartic_spline
 * \ingroup Numerics_Module
 *
 * \brief Primitive function (quartic spline)
 *
 * \tparam DataType data type
 *
 * quartic spline function <br>
 * \f$ 1 - 6 x^2 + 8 x^3 - 3 x^4 \f$
 *
 * Reference: <br>
 * Chen et al., Int. J. Numer. Meth. Eng 2003; 56:935–960.
 */
template <typename DataType>
class quartic_spline : public primitive<DataType, quartic_spline<DataType>>
{
public:
  /*!
   * \brief Primitive function
   *
   * \param  x  input
   * \returns value of \f$ {\mathbf C}(s) = 1 - 6 x^2 + 8 x^3 - 3 x^4 \f$
   */
  inline DataType f(DataType const *x)
  {
    return (*x > static_cast<DataType>(1)) ? DataType{} : 1 + (*x) * (*x) * (-6 + (*x) * (8 - 3 * (*x)));
  }
};

/*! \class cubic_spline
 * \ingroup Numerics_Module
 *
 * \brief Primitive function (cubic spline)
 *
 * \tparam DataType data type
 *
 * cubic spline function <br>
 * \f$
 * {\mathbf C}(s)=\left\{
 * \begin{matrix}
 * 6s^3 -6 s^2 + 1   &\text{for } &s\le \left(\frac{1}{2}\right) \\
 * -2 s^3+6 s^2-6s+2 &\text{for } &\left(\frac{1}{2}\right) < s \le 1 \\
 * 0                 &\text{for } &s > 1
 * \end{matrix}
 * \right.
 * \f$
 *
 * Reference: <br>
 * Chen et al., Int. J. Numer. Meth. Eng 2003; 56:935–960.
 */
template <typename DataType>
class cubic_spline : public primitive<DataType, cubic_spline<DataType>>
{
public:
  /*!
   * \brief Primitive function
   *
   * \param  x  input
   * \returns value of \f$ {\mathbf C}(s) \f$
   */
  inline DataType f(DataType const *x)
  {
    return (*x > static_cast<DataType>(1)) ? DataType{} : (*x > static_cast<DataType>(0.5)) ? 2 - (*x) * (6 - (*x) * (6 - 2 * (*x))) : 1 - (*x) * (*x) * (6 - 6 * (*x));
  }
};

/*! \class normalizedgaussian
 * \ingroup Numerics_Module
 *
 * \brief Primitive function (normalized Gaussian)
 *
 * \tparam DataType data type
 *
 * normalized Gaussian function <br>
 * \f$
 * {\mathbf C}(s)=\left\{
 * \begin{matrix}
 * \frac{e^{-\left(s/\alpha\right)^2} - e^{-\left(1/\alpha\right)^2}}{1-e^{-\left(1/\alpha\right)^2}} &\text{for } &s \le 1 \\
 * 0                                                                                                  &\text{for } &s > 1
 * \end{matrix}
 * \right.
 * \f$
 *
 * Reference: <br>
 * Chen et al., Int. J. Numer. Meth. Eng 2003; 56:935–960.
 */
template <typename DataType>
class normalizedgaussian : public primitive<DataType, normalizedgaussian<DataType>>
{
public:
  /*!
   * \brief Primitive function
   *
   * \param  x  input
   * \returns value of \f$ {\mathbf C}(s) \f$
   */
  inline DataType f(DataType const *x)
  {
    return (*x > static_cast<DataType>(1)) ? DataType{} : (std::exp(-std::pow(*x * alpha, 2)) - std::exp(-alpha * alpha)) / (1 - std::exp(-alpha * alpha));
  }

private:
  //! The \f$ \alpha \f$ parameter is taken to be 0.3, as commonly used in the literature.
  static DataType alpha = static_cast<DataType>(1) / static_cast<DataType>(0.3);
};

#endif // UMUQ_PRIMITIVE
