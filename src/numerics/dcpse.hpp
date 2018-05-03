#ifndef UMHBM_DCPSE_H
#define UMHBM_DCPSE_H

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

class dcpse
{
  public:
  private:
};

#endif
