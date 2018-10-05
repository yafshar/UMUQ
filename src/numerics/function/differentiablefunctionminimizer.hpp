#ifndef UMUQ_DIFFERENTIABLEFUNCTIONMINIMIZER_H
#define UMUQ_DIFFERENTIABLEFUNCTIONMINIMIZER_H

#include "functiontype.hpp"
#include "umuqdifferentiablefunction.hpp"
#include "utilityfunction.hpp"

namespace umuq
{

inline namespace multimin
{

/*!
 * \ingroup Multimin_Module
 * 
 * \brief Different available differentiable Function Minimizer available in UMUQ
 * 
 */
enum differentiableFunctionMinimizerTypes
{
  BFGS = 10,
  BFGS2 = 11,
  CONJUGATEFR = 12,
  CONJUGATEPR = 13,
  STEEPESTDESCENT = 14
};

/*! 
 * \ingroup Multimin_Module
 * 
 * \brief The goal is finding minima of arbitrary multidimensional functions.
 *
 */

/*! \class differentiableFunctionMinimizer
 * \ingroup Multimin_Module
 * 
 * \brief The base class which is for finding minima of arbitrary multidimensional functions
 * with derivative. This is the low level component for a variety of iterative minimizers.
 *
 * This class is the base class for algorithms which require use of the gradient of the function
 * and perform a one-dimensional line minimization along this direction until the lowest point
 * is found to a suitable tolerance.
 * The search direction is then updated with local information from the function and its derivatives,
 * and the whole process repeated until the true n-dimensional minimum is found.
 *
 * \note
 * - It is important to note that the minimization algorithms find local minima; there is
 * no way to determine whether a minimum is a global minimum of the function in question.
 *
 * To use the Minimizer:
 * - First, set the minimizer dimension \sa reset
 * - Second, set the function, input vector and stepsize \sa set
 * - Third, initialize the minimizer \sa init
 * - Forth, iterate until reaching the absolute tolerance \sa iterate
 * 
 * \tparam T Data type
 */
template <typename T>
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
   * \brief Destroy the differentiable Function Minimizer object
   * 
   */
  ~differentiableFunctionMinimizer();

  /*!
   * \brief Move constructor, Construct a new differentiableFunctionMinimizer object
   * 
   * \param other differentiableFunctionMinimizer object
   */
  differentiableFunctionMinimizer(differentiableFunctionMinimizer<T> &&other);

  /*!
   * \brief Move assignment operator
   * 
   */
  differentiableFunctionMinimizer<T> &operator=(differentiableFunctionMinimizer<T> &&other);

  /*!
   * \brief Resizes the minimizer vectors to contain nDim elements
   *
   * \param nDim  New size of the minimizer vectors
   *
   * \returns true
   */
  virtual bool reset(int const nDim) noexcept;

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   * 
   * \param umFun     umuq Differentiable Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>> &umFun, std::vector<T> const &X, T const StepSize, T const Tol);
  virtual bool set(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>> &umFun, T const *X, T const StepSize, T const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   * 
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param DFun      Function gradient \f$ \nabla \f$ to be used in this minimizer
   * \param FDFun     Function & its gradient to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<T> &Fun, DF_MTYPE<T> &DFun, FDF_MTYPE<T> &FDFun, std::vector<T> const &X, T const StepSize, T const Tol);
  virtual bool set(F_MTYPE<T> const &Fun, DF_MTYPE<T> const &DFun, FDF_MTYPE<T> const &FDFun, std::vector<T> const &X, T const StepSize, T const Tol);

  virtual bool set(F_MTYPE<T> &Fun, std::vector<T> const &X, T const StepSize, T const Tol);
  virtual bool set(F_MTYPE<T> const &Fun, std::vector<T> const &X, T const StepSize, T const Tol);

  virtual bool set(F_MTYPE<T> &Fun, DF_MTYPE<T> &DFun, FDF_MTYPE<T> &FDFun, T const *X, T const StepSize, T const Tol);
  virtual bool set(F_MTYPE<T> const &Fun, DF_MTYPE<T> const &DFun, FDF_MTYPE<T> const &FDFun, T const *X, T const StepSize, T const Tol);

  virtual bool set(F_MTYPE<T> &Fun, T const *X, T const StepSize, T const Tol);
  virtual bool set(F_MTYPE<T> const &Fun, T const *X, T const StepSize, T const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   *
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param DFun      Function gradient \f$ \nabla \f$ to be used in this minimizer
   * \param FDFun     Function & its gradient to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<T> &Fun, DF_MTYPE<T> &DFun, FDF_MTYPE<T> &FDFun, std::vector<T> const &Params, std::vector<T> const &X, T const StepSize, T const Tol);
  virtual bool set(F_MTYPE<T> const &Fun, DF_MTYPE<T> const &DFun, FDF_MTYPE<T> const &FDFun, std::vector<T> const &Params, std::vector<T> const &X, T const StepSize, T const Tol);

  virtual bool set(F_MTYPE<T> &Fun, std::vector<T> const &Params, std::vector<T> const &X, T const StepSize, T const Tol);
  virtual bool set(F_MTYPE<T> const &Fun, std::vector<T> const &Params, std::vector<T> const &X, T const StepSize, T const Tol);
  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial StepSize
   *
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param DFun      Function gradient \f$ \nabla \f$ to be used in this minimizer
   * \param FDFun     Function & its gradient to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param NumParams Number of dimensions (Number of parameters of the Function object)
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<T> &Fun, DF_MTYPE<T> &DFun, FDF_MTYPE<T> &FDFun, T const *Params, int const NumParams, T const *X, T const StepSize, T const Tol);
  virtual bool set(F_MTYPE<T> const &Fun, DF_MTYPE<T> const &DFun, FDF_MTYPE<T> const &FDFun, T const *Params, int const NumParams, T const *X, T const StepSize, T const Tol);

  virtual bool set(F_MTYPE<T> &Fun, T const *Params, int const NumParams, T const *X, T const StepSize, T const Tol);
  virtual bool set(F_MTYPE<T> const &Fun, T const *Params, int const NumParams, T const *X, T const StepSize, T const Tol);

  /*!
   * \brief Set the N-dimensional initial vector and initial StepSize
   *
   * \param X         N-dimensional initial vector
   * \param StepSize  N-dimensional initial step size vector
   * \param Tol       The user-supplied tolerance
   *
   * \return true
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(std::vector<T> const &X, T const StepSize, T const Tol);
  virtual bool set(T const *X, T const StepSize, T const Tol);

  /*!
   * \brief Initialize the minimizer
   * 
   * \return true 
   * \return false 
   */
  virtual bool init();

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
  inline std::string const getName() const;

  /*!
   * \brief Get N-dimensional x vector
   *
   * \return T*
   */
  inline T *getX();

  /*!
   * \brief Get N-dimensional dx vector
   *
   * \return T*
   */
  inline T *getdX();

  /*!
   * \brief Get N-dimensional gradient vector
   *
   * \return T*
   */
  inline T *getGradient();

  /*!
   * \brief Helper function to test the norm of the gradient against the absolute tolerance, since the gradient goes to zero at a minimum
   * 
   * \param G       Input gradient vector
   * \param abstol  Absolute tolerance
   *  
   * \return -1, 0, and 1 (where -1:Fail, 0:Success, and 1:Continue)
   */
  int testGradient(T const *G, T const abstol);
  int testGradient(std::vector<T> const &G, T const abstol);

  /*!
   * \brief Get the minimum function value
   *
   * \return the minimum function value
   */
  inline T getMin();

  /*!
   * \brief Get the number of dimensions 
   * 
   * \return /number of dimensions
   */
  inline int getDimension();

  /*!
   * \brief Helper function to compute the gradient of the function f at X (\f$ \frac{\partial f}{\partial x} \f$)
   * 
   * \note 
   * - Helper function to compute the gradient by a finite-difference approximation in one-dimension.
   * - Using this routine is not advised, you should probably use a derivative-free algorithm instead.
   * - Finite-difference approximations are not only expensive, but they are also notoriously susceptible to roundoff 
   * errors. 
   * - On the other hand, finite-difference approximations are very useful to check that your analytical 
   * gradient computation is correct—this is always a good idea, because in my experience it is very easy to have 
   * bugs in your gradient code, and an incorrect gradient will cause weird problems with a gradient-based 
   * optimization algorithm.
   * 
   * \param X  Input point
   * \param G  Gradient of the function f at X (\f$ \frac{\partial f}{\partial x} \f$)
   * 
   * \return true 
   * \return false 
   */
  bool df(T const *X, T *G);

  /*!
   * \brief Helper function to compute the function value, and its gradient at X (\f$ \frac{\partial f}{\partial x} \f$)
   * 
   * \param X  Input point
   * \param F  Function value at X
   * \param G  Function gradient \f$ \nabla \f$ at X
   * 
   * \return true 
   * \return false 
   */
  bool fdf(T const *X, T *F, T *G);

  /*!
   * \brief Compute new trial point at \f$ x - step * p \f$, where p is the current direction
   * 
   * \param X 
   * \param P       Current direction
   * \param Step    step
   * \param lambda  Coefficient
   * \param X1      New trial point
   * \param DX 
   */
  void takeStep(T const *X, T const *P, T const Step, T const lambda, T *X1, T *DX);
  void takeStep(std::vector<T> const &X, std::vector<T> const &P, T const Step, T const lambda, std::vector<T> &X1, std::vector<T> &DX);

  /*!
   * \brief 
   * 
   * \param X 
   * \param P 
   * \param lambda 
   * \param pg 
   * \param stepa 
   * \param stepc 
   * \param fa 
   * \param fc 
   * \param X1 
   * \param DX 
   * \param Gradient 
   * \param Step 
   * \param Fval 
   */
  bool intermediatePoint(T const *X, T const *P,
                         T const lambda, T const pg,
                         T const stepc,
                         T const fa, T const fc,
                         T *X1, T *DX,
                         T *Gradient, T *Step, T *Fval);

  bool intermediatePoint(std::vector<T> const &X, std::vector<T> const &P,
                         T const lambda, T const pg,
                         T const stepc,
                         T const fa, T const fc,
                         std::vector<T> &X1, std::vector<T> &DX,
                         std::vector<T> &Gradient, T &Step, T &Fval);

  /*!
   * \brief   This function starting at \f$ (x0, f0) \f$ move along the direction p to find a minimum
   *          \f$ f(x0 - lambda * p) \f$, returning the new point \f$ x1 = x0-lambda*p, \f$
   *          \f$ f1=f(x1) \f$ and \f$ g1 = grad(f) \f$ at x1
   * 
   * \param X 
   * \param P 
   * \param lambda 
   * \param stepa 
   * \param stepb 
   * \param stepc 
   * \param fa 
   * \param fb 
   * \param fc 
   * \param Tol 
   * \param X1 
   * \param DX1 
   * \param X2 
   * \param DX2 
   * \param Gradient 
   * \param Step 
   * \param Fval 
   * \param Gnorm 
   */
  bool minimize(T const *X, T const *P,
                T const lambda, T const stepa,
                T const stepb, T const stepc,
                T const fa, T const fb,
                T const fc, T const Tol,
                T *X1, T *DX1,
                T *X2, T *DX2,
                T *Gradient, T *Step,
                T *Fval, T *Gnorm);

  bool minimize(std::vector<T> const &X, std::vector<T> const &P,
                T const lambda, T const stepa,
                T const stepb, T const stepc,
                T const fa, T const fb,
                T const fc, T const Tol,
                std::vector<T> &X1, std::vector<T> &DX1,
                std::vector<T> &X2, std::vector<T> &DX2,
                std::vector<T> &Gradient, T &Step,
                T &Fval, T &Gnorm);

private:
  // Make it noncopyable
  differentiableFunctionMinimizer(differentiableFunctionMinimizer<T> const &) = delete;

  // Make it not assignable
  differentiableFunctionMinimizer<T> &operator=(differentiableFunctionMinimizer<T> const &) = delete;

public:
  //! Name of the differentiableFunctionMinimizer
  std::string name;

  // multi dimensional part
  umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>> fun;

  //! N-dimensional x vector
  std::vector<T> x;

  //! N-dimensional dx vector
  std::vector<T> dx;

  //! N-dimensional gradient vector
  std::vector<T> gradient;

  //!
  T step;

  //!
  T maxStep;

  //! Tolerance
  T tol;

  //! Function value
  T fval;
};

template <typename T>
differentiableFunctionMinimizer<T>::differentiableFunctionMinimizer(char const *Name) : name(Name) {}

template <typename T>
differentiableFunctionMinimizer<T>::~differentiableFunctionMinimizer() {}

template <typename T>
differentiableFunctionMinimizer<T>::differentiableFunctionMinimizer(differentiableFunctionMinimizer<T> &&other)
{
  name = other.name;
  fun = std::move(other.fun);
  x = std::move(other.x);
  dx = std::move(other.dx);
  gradient = std::move(other.gradient);
  step = other.step;
  maxStep = other.maxStep;
  tol = other.tol;
  fval = other.fval;
}

template <typename T>
differentiableFunctionMinimizer<T> &differentiableFunctionMinimizer<T>::operator=(differentiableFunctionMinimizer<T> &&other)
{
  name = other.name;
  fun = std::move(other.fun);
  x = std::move(other.x);
  dx = std::move(other.dx);
  gradient = std::move(other.gradient);
  step = other.step;
  maxStep = other.maxStep;
  tol = other.tol;
  fval = other.fval;

  return *this;
}

template <typename T>
bool differentiableFunctionMinimizer<T>::reset(int const nDim) noexcept
{
  if (nDim <= 0)
  {
    UMUQFAILRETURN("Invalid number of parameters specified!");
  }

  x.resize(nDim);
  dx.resize(nDim);
  gradient.resize(nDim);

  return true;
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>> &umFun, std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>> &umFun, T const *X, T const StepSize, T const Tol)
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> &Fun, DF_MTYPE<T> &DFun, FDF_MTYPE<T> &FDFun, std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> const &Fun, DF_MTYPE<T> const &DFun, FDF_MTYPE<T> const &FDFun, std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> &Fun, DF_MTYPE<T> &DFun, FDF_MTYPE<T> &FDFun, T const *X, T const StepSize, T const Tol)
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> const &Fun, DF_MTYPE<T> const &DFun, FDF_MTYPE<T> const &FDFun, T const *X, T const StepSize, T const Tol)
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> &Fun, std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
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

  fun.df = std::bind(&differentiableFunctionMinimizer<T>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<T>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> const &Fun, std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
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

  fun.df = std::bind(&differentiableFunctionMinimizer<T>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<T>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> &Fun, T const *X, T const StepSize, T const Tol)
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

  fun.df = std::bind(&differentiableFunctionMinimizer<T>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<T>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> const &Fun, T const *X, T const StepSize, T const Tol)
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

  fun.df = std::bind(&differentiableFunctionMinimizer<T>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<T>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> &Fun, DF_MTYPE<T> &DFun, FDF_MTYPE<T> &FDFun, std::vector<T> const &Params, std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>>(Params));
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> &Fun, std::vector<T> const &Params, std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  fun.df = std::bind(&differentiableFunctionMinimizer<T>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<T>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> const &Fun, DF_MTYPE<T> const &DFun, FDF_MTYPE<T> const &FDFun, std::vector<T> const &Params, std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>>(Params));
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> const &Fun, std::vector<T> const &Params, std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (X.size() == x.size())
  {
    std::copy(X.begin(), X.end(), x.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  fun.df = std::bind(&differentiableFunctionMinimizer<T>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<T>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> &Fun, DF_MTYPE<T> &DFun, FDF_MTYPE<T> &FDFun, T const *Params, int const NumParams, T const *X, T const StepSize, T const Tol)
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
    fun = std::move(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>>(Params, NumParams));
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> &Fun, T const *Params, int const NumParams, T const *X, T const StepSize, T const Tol)
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
    fun = std::move(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  fun.df = std::bind(&differentiableFunctionMinimizer<T>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<T>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> const &Fun, DF_MTYPE<T> const &DFun, FDF_MTYPE<T> const &FDFun, T const *Params, int const NumParams, T const *X, T const StepSize, T const Tol)
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
    fun = std::move(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>>(Params, NumParams));
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(F_MTYPE<T> const &Fun, T const *Params, int const NumParams, T const *X, T const StepSize, T const Tol)
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
    fun = std::move(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  fun.df = std::bind(&differentiableFunctionMinimizer<T>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<T>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(std::vector<T> const &X, T const StepSize, T const Tol)
{
  if (!fun)
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  if (X.size() != x.size())
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  std::copy(X.begin(), X.end(), x.begin());

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), T{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::set(T const *X, T const StepSize, T const Tol)
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

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::init()
{
  return true;
}

template <typename T>
bool differentiableFunctionMinimizer<T>::iterate()
{
  return true;
}

template <typename T>
bool differentiableFunctionMinimizer<T>::restart()
{
  return true;
}

template <typename T>
inline std::string const differentiableFunctionMinimizer<T>::getName() const
{
  return name;
}

template <typename T>
inline T *differentiableFunctionMinimizer<T>::getX()
{
  return x.data();
}

template <typename T>
inline T *differentiableFunctionMinimizer<T>::getdX()
{
  return dx.data();
}

template <typename T>
inline T *differentiableFunctionMinimizer<T>::getGradient()
{
  return gradient.data();
}

template <typename T>
inline int differentiableFunctionMinimizer<T>::testGradient(T const *G, T const abstol)
{
  if (abstol < T{})
  {
    UMUQWARNING("Absolute tolerance is negative!");
    // fail
    return -1;
  }

  int const n = getDimension();

  // Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
  T norm(0);
  std::for_each(G, G + n, [&](T const g_i) { norm += g_i * g_i; });

  return std::sqrt(norm) >= abstol;
}

template <typename T>
inline int differentiableFunctionMinimizer<T>::testGradient(std::vector<T> const &G, T const abstol)
{
  if (abstol < T{})
  {
    UMUQWARNING("Absolute tolerance is negative!");
    // fail
    return -1;
  }

  // Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
  T norm(0);
  std::for_each(G.begin(), G.end(), [&](T const g_i) { norm += g_i * g_i; });

  return std::sqrt(norm) >= abstol;
}

template <typename T>
inline T differentiableFunctionMinimizer<T>::getMin()
{
  return fval;
}

template <typename T>
inline int differentiableFunctionMinimizer<T>::getDimension()
{
  return x.size();
}

template <typename T>
bool differentiableFunctionMinimizer<T>::df(T const *X, T *G)
{
  if (!G)
  {
    UMUQFAILRETURN("The gradient pointer is not assigned!");
  }

  T const h = std::sqrt(std::numeric_limits<T>::epsilon());

  // Get the number of dimensions
  int const n = getDimension();

  std::vector<T> X1(X, X + n);

  for (int i = 0; i < n; i++)
  {
    T low;
    T high;

    T XI = X[i];

    T DX = std::abs(XI) * h;
    if (DX <= T{})
    {
      DX = h;
    }

    X1[i] = XI + DX;
    high = fun.f(X1.data());

    X1[i] = XI - DX;

    low = fun.f(X1.data());
    X1[i] = XI;

    G[i] = (high - low) / (2. * DX);
  }

  return true;
}

template <typename T>
bool differentiableFunctionMinimizer<T>::fdf(T const *X, T *F, T *G)
{
  if (fun)
  {
    *F = fun.f(X);
    return df(X, G);
  }
  UMUQFAILRETURN("The function is not assigned!");
}

template <typename T>
void differentiableFunctionMinimizer<T>::takeStep(T const *X, T const *P, T const Step, T const lambda, T *X1, T *DX)
{
  int const n = getDimension();

  // Compute the sum \f$y = \alpha x + y\f$ for the vectors x and y (set dx to zero).
  T const alpha = -Step * lambda;

  for (int i = 0; i < n; i++)
  {
    DX[i] = alpha * P[i];
  }

  std::copy(X, X + n, X1);

  for (int i = 0; i < n; i++)
  {
    X1[i] += DX[i];
  }
}

template <typename T>
void differentiableFunctionMinimizer<T>::takeStep(std::vector<T> const &X, std::vector<T> const &P, T const Step, T const lambda, std::vector<T> &X1, std::vector<T> &DX)
{
  int const n = getDimension();

  // Compute the sum \f$y = \alpha x + y\f$ for the vectors x and y (set dx to zero).
  T const alpha = -Step * lambda;

  for (int i = 0; i < n; i++)
  {
    DX[i] = alpha * P[i];
  }

  std::copy(X.begin(), X.end(), X1.begin());

  for (int i = 0; i < n; i++)
  {
    X1[i] += DX[i];
  }
}

template <typename T>
bool differentiableFunctionMinimizer<T>::intermediatePoint(T const *X, T const *P,
                                                           T const lambda, T const pg,
                                                           T const stepc,
                                                           T const fa, T const fc,
                                                           T *X1, T *DX,
                                                           T *Gradient, T *Step, T *Fval)
{
  T stepb(1);
  T stepd(stepc);

  T fb(fa);
  T fd(fc);

  while (fb >= fa && stepb > T{})
  {
    T u = std::abs(pg * lambda * stepd);

    stepb = 0.5 * stepd * u / ((fd - fa) + u);

    takeStep(X, P, stepb, lambda, X1, DX);

    fb = fun.f(X1);

    if (fb >= fa && stepb > T{})
    {
      // Downhill step failed, reduce step-size and try again
      fd = fb;
      stepd = stepb;
    }
  }

  *Step = stepb;

  *Fval = fb;

  return fun.df(X1, Gradient);
}

template <typename T>
bool differentiableFunctionMinimizer<T>::intermediatePoint(std::vector<T> const &X, std::vector<T> const &P,
                                                           T const lambda, T const pg,
                                                           T const stepc,
                                                           T const fa, T const fc,
                                                           std::vector<T> &X1, std::vector<T> &DX,
                                                           std::vector<T> &Gradient, T &Step, T &Fval)
{
  T stepb(1);
  T stepd(stepc);

  T fb(fa);
  T fd(fc);

  while (fb >= fa && stepb > T{})
  {
    T u = std::abs(pg * lambda * stepd);

    stepb = 0.5 * stepd * u / ((fd - fa) + u);

    takeStep(X, P, stepb, lambda, X1, DX);

    fb = fun.f(X1.data());

    if (fb >= fa && stepb > T{})
    {
      // Downhill step failed, reduce step-size and try again
      fd = fb;
      stepd = stepb;
    }
  }

  Step = stepb;

  Fval = fb;

  return fun.df(X1.data(), Gradient.data());
}

template <typename T>
bool differentiableFunctionMinimizer<T>::minimize(T const *X, T const *P,
                                                  T const lambda, T const stepa,
                                                  T const stepb, T const stepc,
                                                  T const fa, T const fb,
                                                  T const fc, T const Tol,
                                                  T *X1, T *DX1,
                                                  T *X2, T *DX2,
                                                  T *Gradient, T *Step,
                                                  T *Fval, T *Gnorm)
{
  // Starting at (x0, f0) move along the direction p to find a minimum
  // \f$ f(x0 - lambda * p) \f$, returning the new point \f$ x1 = x0-lambda*p, \f$
  // \f$ f1=f(x1) \f$ and \f$ g1 = grad(f) \f$ at x1

  int const n = getDimension();

  T stpb(stepb);
  T stpa(stepa);
  T stpc(stepc);

  T fstpb(fb);

  T u(stepb);
  T v(stepa);
  T w(stepc);

  T fu(fb);
  T fv(fa);
  T fw(fc);

  T old2 = std::abs(w - v);
  T old1 = std::abs(v - u);

  T stepm;
  T fm;
  T pg;
  T gnorm1;

  std::copy(X1, X1 + n, X2);
  std::copy(DX1, DX1 + n, DX2);

  *Fval = fstpb;
  *Step = stpb;

  T s(0);
  std::for_each(Gradient, Gradient + n, [&](T const g_i) { s += g_i * g_i; });
  *Gnorm = std::sqrt(s);

  int iter(0);

  // mid_trial:
  for (;;)
  {
    iter++;

    if (iter > 10)
    {
      // MAX ITERATIONS
      return false;
    }

    {
      T dw = w - u;
      T dv = v - u;

      T e1 = ((fv - fu) * dw * dw + (fu - fw) * dv * dv);
      T e2 = 2 * ((fv - fu) * dw + (fu - fw) * dv);

      T du(0);
      if (e2 != T{})
      {
        du = e1 / e2;
      }

      if (du > T{} && du < (stpc - stpb) && std::abs(du) < 0.5 * old2)
      {
        stepm = u + du;
      }
      else if (du < T{} && du > (stpa - stpb) && std::abs(du) < 0.5 * old2)
      {
        stepm = u + du;
      }
      else if ((stpc - stpb) > (stpb - stpa))
      {
        stepm = 0.38 * (stpc - stpb) + stpb;
      }
      else
      {
        stepm = stpb - 0.38 * (stpb - stpa);
      }
    }

    takeStep(X, P, stepm, lambda, X1, DX1);

    fm = fun.f(X1);

#ifdef DEBUG
    std::cout << "Trying stepm = " << stepm << " fm = " << fm << std::endl;
#endif

    if (fm > fstpb)
    {
      if (fm < fv)
      {
        w = v;
        v = stepm;
        fw = fv;
        fv = fm;
      }
      else if (fm < fw)
      {
        w = stepm;
        fw = fm;
      }

      if (stepm < stpb)
      {
        stpa = stepm;
      }
      else
      {
        stpc = stepm;
      }

      // goto mid_trial;
      continue;
    }
    else if (fm <= fstpb)
    {
      old2 = old1;
      old1 = std::abs(u - stepm);
      w = v;
      v = u;
      u = stepm;
      fw = fv;
      fv = fu;
      fu = fm;

      std::copy(X1, X1 + n, X2);
      std::copy(DX1, DX1 + n, DX2);

      fun.df(X1, Gradient);

      pg = T{};
      for (int i = 0; i < n; i++)
      {
        pg += P[i] * Gradient[i];
      }

      s = (T)0;
      std::for_each(Gradient, Gradient + n, [&](T const g_i) { s += g_i * g_i; });
      gnorm1 = std::sqrt(s);

#ifdef DEBUG
      /*!
       * \todo
       * Use IO class to print out p, gradient, pg for debugging purpose
       */
#endif
      *Fval = fm;
      *Step = stepm;
      *Gnorm = gnorm1;

      if (std::abs(pg * lambda / gnorm1) < Tol)
      {
#ifdef DEBUG
        std::cout << "Ok!" << std::endl;
#endif
        // SUCCESS
        return true;
      }

      if (stepm < stpb)
      {
        stpc = stpb;
        stpb = stepm;
        fstpb = fm;
      }
      else
      {
        stpa = stpb;
        stpb = stepm;
        fstpb = fm;
      }

      // Goto mid_trial;
      continue;
    }
  }
}

template <typename T>
bool differentiableFunctionMinimizer<T>::minimize(std::vector<T> const &X, std::vector<T> const &P,
                                                  T const lambda, T const stepa,
                                                  T const stepb, T const stepc,
                                                  T const fa, T const fb,
                                                  T const fc, T const Tol,
                                                  std::vector<T> &X1, std::vector<T> &DX1,
                                                  std::vector<T> &X2, std::vector<T> &DX2,
                                                  std::vector<T> &Gradient, T &Step,
                                                  T &Fval, T &Gnorm)
{
  // Starting at (x0, f0) move along the direction p to find a minimum
  // \f$ f(x0 - lambda * p) \f$, returning the new point \f$ x1 = x0-lambda*p, \f$
  // \f$ f1=f(x1) \f$ and \f$ g1 = grad(f) \f$ at x1

  int const n = getDimension();

  T stpb(stepb);
  T stpa(stepa);
  T stpc(stepc);

  T fstpb(fb);

  T u(stepb);
  T v(stepa);
  T w(stepc);

  T fu(fb);
  T fv(fa);
  T fw(fc);

  T old2 = std::abs(w - v);
  T old1 = std::abs(v - u);

  T stepm;
  T fm;
  T pg;
  T gnorm1;

  std::copy(X1.begin(), X1.end(), X2.begin());
  std::copy(DX1.begin(), DX1.end(), DX2.begin());

  Fval = fstpb;
  Step = stpb;

  T s(0);
  std::for_each(Gradient.begin(), Gradient.end(), [&](T const g_i) { s += g_i * g_i; });
  Gnorm = std::sqrt(s);

  int iter(0);

  // mid_trial:
  for (;;)
  {
    iter++;

    if (iter > 10)
    {
      // MAX ITERATIONS
      return false;
    }

    {
      T dw = w - u;
      T dv = v - u;

      T e1 = ((fv - fu) * dw * dw + (fu - fw) * dv * dv);
      T e2 = 2 * ((fv - fu) * dw + (fu - fw) * dv);

      T du(0);
      if (e2 != T{})
      {
        du = e1 / e2;
      }

      if (du > T{} && du < (stpc - stpb) && std::abs(du) < 0.5 * old2)
      {
        stepm = u + du;
      }
      else if (du < T{} && du > (stpa - stpb) && std::abs(du) < 0.5 * old2)
      {
        stepm = u + du;
      }
      else if ((stpc - stpb) > (stpb - stpa))
      {
        stepm = 0.38 * (stpc - stpb) + stpb;
      }
      else
      {
        stepm = stpb - 0.38 * (stpb - stpa);
      }
    }

    takeStep(X, P, stepm, lambda, X1, DX1);

    fm = fun.f(X1.data());

#ifdef DEBUG
    std::cout << "Trying stepm = " << stepm << " fm = " << fm << std::endl;
#endif

    if (fm > fstpb)
    {
      if (fm < fv)
      {
        w = v;
        v = stepm;
        fw = fv;
        fv = fm;
      }
      else if (fm < fw)
      {
        w = stepm;
        fw = fm;
      }

      if (stepm < stpb)
      {
        stpa = stepm;
      }
      else
      {
        stpc = stepm;
      }

      // goto mid_trial;
      continue;
    }
    else if (fm <= fstpb)
    {
      old2 = old1;
      old1 = std::abs(u - stepm);
      w = v;
      v = u;
      u = stepm;
      fw = fv;
      fv = fu;
      fu = fm;

      std::copy(X1.begin(), X1.end(), X2.begin());
      std::copy(DX1.begin(), DX1.end(), DX2.begin());

      fun.df(X1.data(), Gradient.data());

      pg = T{};
      for (int i = 0; i < n; i++)
      {
        pg += P[i] * Gradient[i];
      }

      s = (T)0;
      std::for_each(Gradient.begin(), Gradient.end(), [&](T const g_i) { s += g_i * g_i; });
      gnorm1 = std::sqrt(s);

#ifdef DEBUG
      /*!
       * \todo
       * Use IO class to print out p, gradient, pg for debugging purpose
       */
#endif
      Fval = fm;
      Step = stepm;
      Gnorm = gnorm1;

      if (std::abs(pg * lambda / gnorm1) < Tol)
      {
#ifdef DEBUG
        std::cout << "Ok!" << std::endl;
#endif
        // SUCCESS
        return true;
      }

      if (stepm < stpb)
      {
        stpc = stpb;
        stpb = stepm;
        fstpb = fm;
      }
      else
      {
        stpa = stpb;
        stpb = stepm;
        fstpb = fm;
      }

      // Goto mid_trial;
      continue;
    }
  }
}

} // namespace multimin
} // namespace umuq

#endif // UMUQ_DIFFERENTIABLEFUNCTIONMINIMIZER
