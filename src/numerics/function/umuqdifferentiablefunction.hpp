#ifndef UMUQ_UMUQDIFFERENTIABLEFUNCTION_H
#define UMUQ_UMUQDIFFERENTIABLEFUNCTION_H

#include "core/core.hpp"
#include "umuqfunction.hpp"

#include <vector>
#include <utility>
#include <functional>

namespace umuq
{

/*!\class umuqDifferentiableFunction
 * \brief umuqDifferentiableFunction is a general-purpose polymorphic differentiable function wrapper of n variables
 *
 * \tparam DataType                       Data type
 * \tparam FunctionType                   Function type (wrapped as std::function)
 * \tparam DerivativeFunctionType         Derivative Function type (wrapped as std::function)
 * \tparam FunctionDerivativeFunctionType Function & Derivative Function type (wrapped as std::function)
 */
template <typename DataType,
          class FunctionType,
          class DerivativeFunctionType = FunctionType,
          class FunctionDerivativeFunctionType = std::function<void(DataType const *, DataType const *, DataType *, DataType *)>>
class umuqDifferentiableFunction : public umuqFunction<DataType, FunctionType>
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
    umuqDifferentiableFunction(DataType const *Params, int const NumParams, char const *Name = "");

    /*!
     * \brief Construct a new umuqDifferentiableFunction object
     *
     * \param Params  Input parameters of the Function object
     * \param Name    Function name
     */
    umuqDifferentiableFunction(std::vector<DataType> const &Params, char const *Name = "");

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
    umuqDifferentiableFunction(umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType> &&other);

    /*!
     * \brief Move assignment operator
     *
     */
    umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType> &operator=(umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType> &&other);

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
     * Avoiding implicit generation of the copy constructor.
     */
    umuqDifferentiableFunction(umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType> const &) = delete;

    /*!
     * \brief Delete a umuqDifferentiableFunction object assignment
     *
     * Avoiding implicit copy assignment.
     *
     * \returns umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType>&
     */
    umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType> &operator=(umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType> const &) = delete;

  public:
    /*!
     * \brief A general-purpose polymorphic function wrapper which calculates the gradient of the function. \sa umuq::umuqFunction::f.
     *
     * Computes the gradient of the function (it computes the n-dimensional gradient \f$ \nabla {f} = \frac{\partial f(x)}{\partial x_i} \f$)
     */
    DerivativeFunctionType df;

    /*!
     * \brief A general-purpose polymorphic function wrapper which calculates both the function value and it's derivative together.
     *
     * It uses a provided parametric function of n variables to operate on and also
     * a function which calculates the gradient of the function. <br>
     * It is faster to compute the function and its derivative at the same time.
     */
    FunctionDerivativeFunctionType fdf;
};

template <typename DataType, class FunctionType, class DerivativeFunctionType, class FunctionDerivativeFunctionType>
umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType>::umuqDifferentiableFunction(char const *Name) : umuqFunction<DataType, FunctionType>(Name),
                                                                                                                                                           df(nullptr),
                                                                                                                                                           fdf(nullptr) {}

template <typename DataType, class FunctionType, class DerivativeFunctionType, class FunctionDerivativeFunctionType>
umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType>::umuqDifferentiableFunction(int const nDim, char const *Name) : umuqFunction<DataType, FunctionType>(nDim, Name),
                                                                                                                                                                           df(nullptr),
                                                                                                                                                                           fdf(nullptr) {}

template <typename DataType, class FunctionType, class DerivativeFunctionType, class FunctionDerivativeFunctionType>
umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType>::umuqDifferentiableFunction(DataType const *Params, int const NumParams, char const *Name) : umuqFunction<DataType, FunctionType>(Params, NumParams, Name),
                                                                                                                                                                                                        df(nullptr),
                                                                                                                                                                                                        fdf(nullptr) {}

template <typename DataType, class FunctionType, class DerivativeFunctionType, class FunctionDerivativeFunctionType>
umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType>::umuqDifferentiableFunction(std::vector<DataType> const &Params, char const *Name) : umuqFunction<DataType, FunctionType>(Params, Name),
                                                                                                                                                                                                df(nullptr),
                                                                                                                                                                                                fdf(nullptr) {}

template <typename DataType, class FunctionType, class DerivativeFunctionType, class FunctionDerivativeFunctionType>
umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType>::~umuqDifferentiableFunction() {}

template <typename DataType, class FunctionType, class DerivativeFunctionType, class FunctionDerivativeFunctionType>
umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType>::umuqDifferentiableFunction(umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType> &&other) : umuqDifferentiableFunction<DataType, FunctionType>(std::move(other)),
                                                                                                                                                                                                                                                             df(std::move(other.df)),
                                                                                                                                                                                                                                                             fdf(std::move(other.fdf))
{
}

template <typename DataType, class FunctionType, class DerivativeFunctionType, class FunctionDerivativeFunctionType>
umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType> &umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType>::operator=(umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType> &&other)
{
    umuqFunction<DataType, FunctionType>::operator=(std::move(other));
    df = std::move(other.df);
    fdf = std::move(other.fdf);
    return *this;
}

template <typename DataType, class FunctionType, class DerivativeFunctionType, class FunctionDerivativeFunctionType>
umuqDifferentiableFunction<DataType, FunctionType, DerivativeFunctionType, FunctionDerivativeFunctionType>::operator bool() const noexcept
{
    return (this->f != nullptr && df != nullptr && fdf != nullptr);
}

} // namespace umuq

#endif // UMUQ_UMUQDIFFERENTIABLEFUNCTION
