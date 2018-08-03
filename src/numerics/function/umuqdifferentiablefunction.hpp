#ifndef UMUQ_UMUQDIFFERENTIABLEFUNCTION_H
#define UMUQ_UMUQDIFFERENTIABLEFUNCTION_H

#include "umuqfunction.hpp"

/*!\class umuqDifferentiableFunction
 * \brief umuqDifferentiableFunction is a general-purpose polymorphic differentiable function wrapper of n variables
 *
 * \tparam T  Data type
 * \tparam F  Function type
 * \tparam D  Function Derivative type 
 */
template <typename T, class F, class D = F>
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
     * \brief Destroy the umuq Differentiable Function object
     * 
     */
    ~umuqDifferentiableFunction();

    /*!
     * \brief Move constructor, Construct a new umuqDifferentiableFunction object
     * 
     * \param other umuqDifferentiableFunction object
     */
    umuqDifferentiableFunction(umuqDifferentiableFunction<T, F, D> &&other);

    /*!
     * \brief Move assignment operator
     * 
     */
    umuqDifferentiableFunction<T, F, D> &operator=(umuqDifferentiableFunction<T, F, D> &&other);

  private:
    //! Make it noncopyable
    umuqDifferentiableFunction(umuqDifferentiableFunction<T, F, D> const &) = delete;

    //! Make it not assignable
    umuqDifferentiableFunction<T, F, D> &operator=(umuqDifferentiableFunction<T, F, D> const &) = delete;

  public:
    /*!
     * \brief A general-purpose polymorphic function wrapper (for a derivative function)
     * 
     * \returns the value of the function 
     */
    D df;
};

template <typename T, class F, class D>
umuqDifferentiableFunction<T, F, D>::umuqDifferentiableFunction(char const *Name) : umuqFunction<T, F>(Name) {}

template <typename T, class F, class D>
umuqDifferentiableFunction<T, F, D>::umuqDifferentiableFunction(int const nDim, char const *Name) : umuqFunction<T, F>(nDim, Name) {}

template <typename T, class F, class D>
umuqDifferentiableFunction<T, F, D>::~umuqDifferentiableFunction() {}

template <typename T, class F, class D>
umuqDifferentiableFunction<T, F, D>::umuqDifferentiableFunction(umuqDifferentiableFunction<T, F, D> &&other) : umuqDifferentiableFunction<T, F>(std::move(other)),
                                                                                                               df(std::move(other.df))
{
}

template <typename T, class F, class D>
umuqDifferentiableFunction<T, F, D> &umuqDifferentiableFunction<T, F, D>::operator=(umuqDifferentiableFunction<T, F, D> &&other)
{
    umuqFunction<T, F>::operator=(std::move(other));
    df = std::move(other.df);

    return *this;
}

#endif // UMUQ_FUNCTION_H
