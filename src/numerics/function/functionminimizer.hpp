#ifndef UMUQ_FUNCTIONMINIMIZER_H
#define UMUQ_FUNCTIONMINIMIZER_H

#include "umuqdifferentiablefunction.hpp"

/*! \brief The goal is finding minima of arbitrary multidimensional functions.
 *  \ingroup multimin_Module
 */

/*! \class functionMinimizer
 * \brief The base class which is for finding minima of arbitrary multidimensional functions.
 * This is the low level component for a variety of iterative minimizers
 * 
 * This class is the base class for algorithms which do not require the gradient of the function. 
 * For example, the Nelder-Mead Simplex algorithm.
 * 
 * NOTE:
 * It is important to note that the minimization algorithms find local minima; there is 
 * no way to determine whether a minimum is a global minimum of the function in question.
 * 
 * \tparam T Data type
 * \tparam F Function type (wrapped as std::function)
 */
template <typename T, class F>
class functionMinimizer
{
public:
  /*!
   * \brief Construct a new function Minimizer object
   * 
   * \param Name Multidimensional function minimizer name
   */
  explicit functionMinimizer(char const *Name = "");

  /*!
   * \brief Resizes the x-vector to contain nDim elements 
   * 
   * \param nDim  New size of the x-vector
   * 
   * \returns true  
   */
  bool reset(int const nDim) noexcept;

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param umFun     umuq Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(umuqFunction<T, F> &umFun, std::vector<T> const &X, std::vector<T> const &stepSize);
  virtual bool set(umuqFunction<T, F> &umFun, T const *X, T const *stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param Fun       Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F &Fun, std::vector<T> const &X, std::vector<T> const &stepSize);
  virtual bool set(F const &Fun, std::vector<T> const &X, std::vector<T> const &stepSize);

  virtual bool set(F &Fun, T const *X, T const *stepSize);
  virtual bool set(F const &Fun, T const *X, T const *stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   *
   * \param Fun       Function to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F &Fun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize);
  virtual bool set(F const &Fun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   *
   * \param Fun       Function to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param NumParams Number of dimensions (Number of parameters of the Function object)
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F &Fun, T const *Params, int const NumParams, T const *X, T const *stepSize);
  virtual bool set(F const &Fun, T const *Params, int const NumParams, T const *X, T const *stepSize);

  /*!
   * \brief Set the N-dimensional initial vector and initial stepSize
   *
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(std::vector<T> const &X, std::vector<T> const &stepSize);
  virtual bool set(T const *X, T const *stepSize);

  /*!
   * \brief Drives the iteration of each algorithm
   * 
   * It performs one iteration to update the state of the minimizer.
   * 
   * \return true 
   * \return false If the iteration encounters an unexpected problem
   */
  virtual bool iterate();

  /*!
   * \brief Get the Name object
   * 
   * \return std::string const 
   */
  std::string const getName() const;

  /*!
   * \brief Return minimizer-specific characteristic size
   * 
   * \returns minimizer-specific characteristic size
   */
  T const getSize() const;

  /*!
   * \brief Get the N-dimensional x vector
   * 
   * \return T* 
   */
  T *getX();

  /*!
   * \brief Get the minimum function value
   * 
   * \return the minimum function value
   */
  T getMin();

private:
  // Make it noncopyable
  functionMinimizer(functionMinimizer<T, F> const &) = delete;

  // Make it not assignable
  functionMinimizer<T, F> &operator=(functionMinimizer<T, F> const &) = delete;

public:
  //! Name of the functionMinimizer
  std::string name;

  //! Function to be used in this minimizer
  umuqFunction<T, F> fun;

  //! N-dimensional x vector
  std::vector<T> x;

  //! The minimizer-specific characteristic size (This size can be used as a stopping criteria)
  T size;

  //! Minimum function value
  T fval;
};

template <typename T, class F>
functionMinimizer<T, F>::functionMinimizer(char const *Name) : name(Name) {}

template <typename T, class F>
bool functionMinimizer<T, F>::reset(int const nDim) noexcept
{
  x.resize(nDim);
  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(umuqFunction<T, F> &umFun, std::vector<T> const &X, std::vector<T> const &stepSize)
{
  if (X.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (umFun)
  {
    fun = std::move(umFun);
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X.begin(), X.end(), x.begin());

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(umuqFunction<T, F> &umFun, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (umFun)
  {
    fun = std::move(umFun);
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(F &Fun, std::vector<T> const &X, std::vector<T> const &stepSize)
{
  if (X.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (Fun)
  {
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X.begin(), X.end(), x.begin());

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(F const &Fun, std::vector<T> const &X, std::vector<T> const &stepSize)
{
  if (X.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (Fun)
  {
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X.begin(), X.end(), x.begin());

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(F &Fun, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(F const &Fun, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(F &Fun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize)
{
  if (X.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqFunction<T, F>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X.begin(), X.end(), x.begin());

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(F const &Fun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize)
{
  if (X.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqFunction<T, F>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X.begin(), X.end(), x.begin());

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(F &Fun, T const *Params, int const NumParams, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqFunction<T, F>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(F const &Fun, T const *Params, int const NumParams, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqFunction<T, F>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(std::vector<T> const &X, std::vector<T> const &stepSize)
{
  if (!fun)
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (X.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  std::copy(X.begin(), X.end(), x.begin());

  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::set(T const *X, T const *stepSize)
{
  if (!fun)
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X, X + x.size(), x.begin());
  return true;
}

template <typename T, class F>
bool functionMinimizer<T, F>::iterate()
{
  return true;
}

template <typename T, class F>
std::string const functionMinimizer<T, F>::getName() const
{
  return name;
}

template <typename T, class F>
T const functionMinimizer<T, F>::getSize() const
{
  return size;
}

template <typename T, class F>
T *functionMinimizer<T, F>::getX()
{
  return x.data();
}

template <typename T, class F>
T functionMinimizer<T, F>::getMin()
{
  return fval;
}

/*! \class differentiableFunctionMinimizer
 * \brief The base class which is for finding minima of arbitrary multidimensional functions
 * with derivative. This is the low level component for a variety of iterative minimizers.
 *
 * This class is the base class for algorithms which require use of the gradient of the function
 * and perform a one-dimensional line minimisation along this direction until the lowest point
 * is found to a suitable tolerance.
 * The search direction is then updated with local information from the function and its derivatives,
 * and the whole process repeated until the true n-dimensional minimum is found.
 *
 * NOTE:
 * It is important to note that the minimization algorithms find local minima; there is
 * no way to determine whether a minimum is a global minimum of the function in question.
 *
 * \tparam T Data type
 * \tparam F Function type (wrapped as std::function)
 * \tparam D Function derivative type (wrapped as std::function)
 */
template <typename T, class F, class D = F, class FD = std::function<void(T const *, T const *, T *, T *)>>
class differentiableFunctionMinimizer
{
public:
  /*!
   * \brief Construct a new differentiable Function Minimizer object
   *
   * \param Name Function name
   */
  explicit differentiableFunctionMinimizer(char const *Name = "");

  /*!
   * \brief Resizes the minimizer vectors to contain nDim elements
   *
   * \param nDim  New size of the minimizer vectors
   *
   * \returns true
   */
  bool reset(int const nDim) noexcept;

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param umFun     umuq Differentiable Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * \param tol       The user-supplied tolerance
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(umuqDifferentiableFunction<T, F, D, FD> &umFun, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol);
  virtual bool set(umuqDifferentiableFunction<T, F, D, FD> &umFun, T const *X, T const *stepSize, T const tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param Fun       Function to be used in this minimizer
   * \param DFun      Function to be used in this minimizer
   * \param FDFun     Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * \param tol       The user-supplied tolerance
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F &Fun, D &DFun, FD &FDFun, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol);
  virtual bool set(F const &Fun, D const &DFun, FD const &FDFun, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol);

  virtual bool set(F &Fun, D &DFun, FD &FDFun, T const *X, T const *stepSize, T const tol);
  virtual bool set(F const &Fun, D const &DFun, FD const &FDFun, T const *X, T const *stepSize, T const tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   *
   * \param Fun       Function to be used in this minimizer
   * \param DFun      Function to be used in this minimizer
   * \param FDFun     Function to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * \param tol       The user-supplied tolerance
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F &Fun, D &DFun, FD &FDFun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol);
  virtual bool set(F const &Fun, D const &DFun, FD const &FDFun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   *
   * \param Fun       Function to be used in this minimizer
   * \param DFun      Function to be used in this minimizer
   * \param FDFun     Function to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param NumParams Number of dimensions (Number of parameters of the Function object)
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * \param tol       The user-supplied tolerance
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F &Fun, D &DFun, FD &FDFun, T const *Params, int const NumParams, T const *X, T const *stepSize, T const tol);
  virtual bool set(F const &Fun, D const &DFun, FD const &FDFun, T const *Params, int const NumParams, T const *X, T const *stepSize, T const tol);

  /*!
   * \brief Set the N-dimensional initial vector and initial stepSize
   *
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * \param tol       The user-supplied tolerance
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(std::vector<T> const &X, std::vector<T> const &stepSize, T const tol);
  virtual bool set(T const *X, T const *stepSize, T const tol);

  /*!
   * \brief Drives the iteration of each algorithm
   *
   * It performs one iteration to update the state of the minimizer.
   *
   * \return true
   * \return false If the iteration encounters an unexpected problem
   */
  virtual bool iterate();

  /*!
   * \brief This function resets the minimizer to use the current point as a new starting point
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool restart();

  /*!
   * \brief Get the Name object
   *
   * \return std::string const
   */
  std::string const getName() const;

  /*!
   * \brief Get N-dimensional x vector
   *
   * \return T*
   */
  T *getX();

  /*!
   * \brief Get N-dimensional dx vector
   *
   * \return T*
   */
  T *getdX();

  /*!
   * \brief Get N-dimensional gradient vector
   *
   * \return T*
   */
  T *getGradient();

  /*!
   * \brief Get the minimum function value
   *
   * \return the minimum function value
   */
  T getMin();

private:
  // Make it noncopyable
  differentiableFunctionMinimizer(differentiableFunctionMinimizer<T, F, D> const &) = delete;

  // Make it not assignable
  differentiableFunctionMinimizer<T, F, D> &operator=(differentiableFunctionMinimizer<T, F, D> const &) = delete;

public:
  //! Name of the differentiableFunctionMinimizer
  std::string name;

  // multi dimensional part
  umuqDifferentiableFunction<T, F, D> fun;

  //! N-dimensional x vector
  std::vector<T> x;

  //! N-dimensional dx vector
  std::vector<T> dx;

  //! N-dimensional gradient vector
  std::vector<T> gradient;

  //! Function value
  T fval;
};

template <typename T, class F, class D, class FD>
differentiableFunctionMinimizer<T, F, D, FD>::differentiableFunctionMinimizer(char const *Name) : name(Name) {}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::reset(int const nDim) noexcept
{
  x.resize(nDim);
  dx.resize(nDim);
  gradient.resize(nDim);

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(umuqDifferentiableFunction<T, F, D, FD> &umFun, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (umFun)
  {
    fun = std::move(umFun);
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(umuqDifferentiableFunction<T, F, D, FD> &umFun, T const *X, T const *stepSize, T const tol)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (umFun)
  {
    fun = std::move(umFun);
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(F &Fun, D &DFun, FD &FDFun, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (Fun)
  {
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (DFun)
  {
    fun.df = DFun;
  }
  else
  {
    UMUQFAILRETURN("Derivative Function is not assigned!");
  }

  if (FDFun)
  {
    fun.fdf = FDFun;
  }
  else
  {
    UMUQFAILRETURN("fdf Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(F const &Fun, D const &DFun, FD const &FDFun, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (Fun)
  {
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (DFun)
  {
    fun.df = DFun;
  }
  else
  {
    UMUQFAILRETURN("Derivative Function is not assigned!");
  }

  if (FDFun)
  {
    fun.fdf = FDFun;
  }
  else
  {
    UMUQFAILRETURN("fdf Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(F &Fun, D &DFun, FD &FDFun, T const *X, T const *stepSize, T const tol)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (DFun)
  {
    fun.df = DFun;
  }
  else
  {
    UMUQFAILRETURN("Derivative Function is not assigned!");
  }

  if (FDFun)
  {
    fun.fdf = FDFun;
  }
  else
  {
    UMUQFAILRETURN("fdf Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(F const &Fun, D const &DFun, FD const &FDFun, T const *X, T const *stepSize, T const tol)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (DFun)
  {
    fun.df = DFun;
  }
  else
  {
    UMUQFAILRETURN("Derivative Function is not assigned!");
  }

  if (FDFun)
  {
    fun.fdf = FDFun;
  }
  else
  {
    UMUQFAILRETURN("fdf Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(F &Fun, D &DFun, FD &FDFun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqDifferentiableFunction<T, F, D, FD>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (DFun)
  {
    fun.df = DFun;
  }
  else
  {
    UMUQFAILRETURN("Derivative Function is not assigned!");
  }

  if (FDFun)
  {
    fun.fdf = FDFun;
  }
  else
  {
    UMUQFAILRETURN("fdf Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(F const &Fun, D const &DFun, FD const &FDFun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize, T const tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqDifferentiableFunction<T, F, D, FD>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (DFun)
  {
    fun.df = DFun;
  }
  else
  {
    UMUQFAILRETURN("Derivative Function is not assigned!");
  }

  if (FDFun)
  {
    fun.fdf = FDFun;
  }
  else
  {
    UMUQFAILRETURN("fdf Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(F &Fun, D &DFun, FD &FDFun, T const *Params, int const NumParams, T const *X, T const *stepSize, T const tol)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqDifferentiableFunction<T, F, D, FD>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (DFun)
  {
    fun.df = DFun;
  }
  else
  {
    UMUQFAILRETURN("Derivative Function is not assigned!");
  }

  if (FDFun)
  {
    fun.fdf = FDFun;
  }
  else
  {
    UMUQFAILRETURN("fdf Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(F const &Fun, D const &DFun, FD const &FDFun, T const *Params, int const NumParams, T const *X, T const *stepSize, T const tol)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqDifferentiableFunction<T, F, D, FD>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (DFun)
  {
    fun.df = DFun;
  }
  else
  {
    UMUQFAILRETURN("Derivative Function is not assigned!");
  }

  if (FDFun)
  {
    fun.fdf = FDFun;
  }
  else
  {
    UMUQFAILRETURN("fdf Function is not assigned!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(std::vector<T> const &X, std::vector<T> const &stepSize, T const tol)
{
  if (!fun)
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (X.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (stepSize.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input step size vector with solver size!");
  }

  std::copy(X.begin(), X.end(), x.begin());

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::set(T const *X, T const *stepSize, T const tol)
{
  if (!fun)
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::iterate()
{
  return true;
}

template <typename T, class F, class D, class FD>
bool differentiableFunctionMinimizer<T, F, D, FD>::restart()
{
  return true;
}

template <typename T, class F, class D, class FD>
std::string const differentiableFunctionMinimizer<T, F, D, FD>::getName() const
{
  return name;
}

template <typename T, class F, class D, class FD>
T *differentiableFunctionMinimizer<T, F, D, FD>::getX()
{
  return x.data();
}

template <typename T, class F, class D, class FD>
T *differentiableFunctionMinimizer<T, F, D, FD>::getdX()
{
  return dx.data();
}

template <typename T, class F, class D, class FD>
T *differentiableFunctionMinimizer<T, F, D, FD>::getGradient()
{
  return gradient.data();
}

template <typename T, class F, class D, class FD>
T differentiableFunctionMinimizer<T, F, D, FD>::getMin()
{
  return fval;
}

#endif //UMUQ_FUNCTIONMINIMIZER_H
