#ifndef UMUQ_UMUQDIFFERENTIABLEFUNCTION_H
#define UMUQ_UMUQDIFFERENTIABLEFUNCTION_H

#include "umuqfunction.hpp"

namespace umuq
{

/*!\class umuqDifferentiableFunction
 * \brief umuqDifferentiableFunction is a general-purpose polymorphic differentiable function wrapper of n variables
 *
 * \tparam T  Data type
 * \tparam F  Function type (wrapped as std::function)
 * \tparam D  Function Derivative type (wrapped as std::function)
 * \tparam FD Function & Derivative type (wrapped as std::function)
 */
template <typename T, class F, class D = F, class FD = std::function<void(T const *, T const *, T *, T *)>>
class umuqDifferentiableFunction : public umuqFunction<T, F>
{
public:
  /*!
   * \brief Construct a new umuqDifferentiableFunction object
   * 
   * \param Name  Function name
   */
  explicit umuqDifferentiableFunction(char const *Name = "");

  /*!
   * \brief Construct a new umuqDifferentiableFunction object
   * 
   * \param nDim  Number of dimensions (Number of parameters) 
   * \param Name  Function name
   */
  umuqDifferentiableFunction(int const nDim, char const *Name = "");

  /*!
   * \brief Construct a new umuqDifferentiableFunction object
   * 
   * \param Params    Input parameters of the Function object
   * \param NumParams Number of dimensions (Number of parameters) 
   * \param Name      Function name
   */
  umuqDifferentiableFunction(T const *Params, int const NumParams, char const *Name = "");

  /*!
   * \brief Construct a new umuqDifferentiableFunction object
   *  
   * \param Params  Input parameters of the Function object
   * \param Name    Function name
   */
  umuqDifferentiableFunction(std::vector<T> const &Params, char const *Name = "");

  /*!
   * \brief Destroy the umuq Differentiable Function object
   * 
   */
  ~umuqDifferentiableFunction();

  /*!
   * \brief Move constructor, Construct a new umuqDifferentiableFunction object
   * 
   * \param other umuqDifferentiableFunction object
   */
  umuqDifferentiableFunction(umuqDifferentiableFunction<T, F, D, FD> &&other);

  /*!
   * \brief Move assignment operator
   * 
   */
  umuqDifferentiableFunction<T, F, D, FD> &operator=(umuqDifferentiableFunction<T, F, D, FD> &&other);

  /*!
   * \brief Checks whether *this stores a callable function target, i.e. is not empty. 
   * 
   * \return true   If it stores a callable function target at f
   * \return false 
   */
  explicit operator bool() const noexcept;

protected:
  /*!
   * \brief Delete an umuqDifferentiableFunction object copy construction
   * 
   * Make it noncopyable.
   */
  umuqDifferentiableFunction(umuqDifferentiableFunction<T, F, D, FD> const &) = delete;

  /*!
   * \brief Delete a umuqDifferentiableFunction object assignment
   * 
   * Make it nonassignable
   * 
   * \returns umuqDifferentiableFunction<T, F, D, FD>& 
   */  
  umuqDifferentiableFunction<T, F, D, FD> &operator=(umuqDifferentiableFunction<T, F, D, FD> const &) = delete;

public:
  /*!
   * \brief A general-purpose polymorphic function wrapper which calculates the gradient of the function. \sa umuq::umuqFunction::f.
   * 
   * Computes the gradient of the function (it computes the n-dimensional gradient \f$ \nabla {f} = \frac{\partial f(x)}{\partial x_i} \f$)
   */
  D df;

  /*!
   * \brief A general-purpose polymorphic function wrapper which calculates both the function value and it's derivative together.
   *  
   * It uses a provided parametric function of n variables to operate on and also 
   * a function which calculates the gradient of the function. <br>
   * It is faster to compute the function and its derivative at the same time.
   */
  FD fdf;
};

template <typename T, class F, class D, class FD>
umuqDifferentiableFunction<T, F, D, FD>::umuqDifferentiableFunction(char const *Name) : umuqFunction<T, F>(Name),
                                                                                        df(nullptr),
                                                                                        fdf(nullptr) {}

template <typename T, class F, class D, class FD>
umuqDifferentiableFunction<T, F, D, FD>::umuqDifferentiableFunction(int const nDim, char const *Name) : umuqFunction<T, F>(nDim, Name),
                                                                                                        df(nullptr),
                                                                                                        fdf(nullptr) {}

template <typename T, class F, class D, class FD>
umuqDifferentiableFunction<T, F, D, FD>::umuqDifferentiableFunction(T const *Params, int const NumParams, char const *Name) : umuqFunction<T, F>(Params, NumParams, Name),
                                                                                                                              df(nullptr),
                                                                                                                              fdf(nullptr) {}

template <typename T, class F, class D, class FD>
umuqDifferentiableFunction<T, F, D, FD>::umuqDifferentiableFunction(std::vector<T> const &Params, char const *Name) : umuqFunction<T, F>(Params, Name),
                                                                                                                      df(nullptr),
                                                                                                                      fdf(nullptr) {}

template <typename T, class F, class D, class FD>
umuqDifferentiableFunction<T, F, D, FD>::~umuqDifferentiableFunction() {}

template <typename T, class F, class D, class FD>
umuqDifferentiableFunction<T, F, D, FD>::umuqDifferentiableFunction(umuqDifferentiableFunction<T, F, D, FD> &&other) : umuqDifferentiableFunction<T, F>(std::move(other)),
                                                                                                                       df(std::move(other.df)),
                                                                                                                       fdf(std::move(other.fdf))
{
}

template <typename T, class F, class D, class FD>
umuqDifferentiableFunction<T, F, D, FD> &umuqDifferentiableFunction<T, F, D, FD>::operator=(umuqDifferentiableFunction<T, F, D, FD> &&other)
{
  umuqFunction<T, F>::operator=(std::move(other));
  df = std::move(other.df);
  fdf = std::move(other.fdf);

  return *this;
}

template <typename T, class F, class D, class FD>
umuqDifferentiableFunction<T, F, D, FD>::operator bool() const noexcept
{
  return (this->f != nullptr && df != nullptr && fdf != nullptr);
}

} // namespace umuq

#endif // UMUQ_UMUQDIFFERENTIABLEFUNCTION
