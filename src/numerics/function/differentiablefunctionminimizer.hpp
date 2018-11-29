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
 * \enum differentiableFunctionMinimizerTypes
 * \ingroup Multimin_Module
 * 
 * \brief Different available differentiable Function Minimizer available in %UMUQ
 * 
 */
enum differentiableFunctionMinimizerTypes
{
  /*! \link umuq::multimin::bfgs The Limited memory Broyden-Fletcher-Goldfarb-Shanno method. */
  BFGS = 10,
  /*! \link umuq::multimin::bfgs2 The Limited memory Broyden-Fletcher-Goldfarb-Shanno method (Fletcher's implementation). */
  BFGS2 = 11,
  /*! \link umuq::multimin::conjugateFr The conjugate gradient Fletcher-Reeve algorithm. */
  CONJUGATEFR = 12,
  /*! \link umuq::multimin::conjugatePr The conjugate Polak-Ribiere gradient algorithm. */
  CONJUGATEPR = 13,
  /*! \link umuq::multimin::steepestDescent The steepestDescent for differentiable function minimizer type. */
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
 * is found to a suitable tolerance.<br>
 * The search direction is then updated with local information from the function and its derivatives,
 * and the whole process repeated until the true n-dimensional minimum is found.
 *
 * \note
 * - It is important to note that the minimization algorithms find local minima; there is
 *   no way to determine whether a minimum is a global minimum of the function in question.
 *
 * To use the Minimizer: <br>
 * - First, set the minimizer dimension \sa reset
 * - Second, set the function, input vector and stepsize \sa set
 * - Third, initialize the minimizer \sa init
 * - Forth, iterate until reaching the absolute tolerance \sa iterate
 * 
 * \tparam DataType Data type
 */
template <typename DataType>
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
  differentiableFunctionMinimizer(differentiableFunctionMinimizer<DataType> &&other);

  /*!
   * \brief Move assignment operator
   * 
   */
  differentiableFunctionMinimizer<DataType> &operator=(differentiableFunctionMinimizer<DataType> &&other);

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
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>> &umFun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   * 
   * \param umFun     umuq Differentiable Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>> &umFun, DataType const *X, DataType const StepSize, DataType const Tol);

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
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, DF_MTYPE<DataType> &DFun, FDF_MTYPE<DataType> &FDFun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

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
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, DF_MTYPE<DataType> const &DFun, FDF_MTYPE<DataType> const &FDFun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   * 
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   * 
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

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
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, DF_MTYPE<DataType> &DFun, FDF_MTYPE<DataType> &FDFun, DataType const *X, DataType const StepSize, DataType const Tol);

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
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, DF_MTYPE<DataType> const &DFun, FDF_MTYPE<DataType> const &FDFun, DataType const *X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   * 
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, DataType const *X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   * 
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, DataType const *X, DataType const StepSize, DataType const Tol);

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
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, DF_MTYPE<DataType> &DFun, FDF_MTYPE<DataType> &FDFun, std::vector<DataType> const &Params, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

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
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, DF_MTYPE<DataType> const &DFun, FDF_MTYPE<DataType> const &FDFun, std::vector<DataType> const &Params, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   *
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param Params    Input parameters of the Function object
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   *
   * \return false If it encounters an unexpected problem 
   */
  virtual bool set(F_MTYPE<DataType> &Fun, std::vector<DataType> const &Params, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and step-size
   *
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param Params    Input parameters of the Function object
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   *
   * \return false If it encounters an unexpected problem 
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, std::vector<DataType> const &Params, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

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
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, DF_MTYPE<DataType> &DFun, FDF_MTYPE<DataType> &FDFun, DataType const *Params, int const NumParams, DataType const *X, DataType const StepSize, DataType const Tol);

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
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, DF_MTYPE<DataType> const &DFun, FDF_MTYPE<DataType> const &FDFun, DataType const *Params, int const NumParams, DataType const *X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial StepSize
   *
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param Params    Input parameters of the Function object
   * \param NumParams Number of dimensions (Number of parameters of the Function object)
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, DataType const *Params, int const NumParams, DataType const *X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial StepSize
   *
   * \param Fun       Function to be used in this minimizer \f$ f(x) \f$
   * \param Params    Input parameters of the Function object
   * \param NumParams Number of dimensions (Number of parameters of the Function object)
   * \param X         N-dimensional initial vector
   * \param StepSize  Step-size
   * \param Tol       The user-supplied tolerance
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, DataType const *Params, int const NumParams, DataType const *X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the N-dimensional initial vector and initial StepSize
   *
   * \param X         N-dimensional initial vector
   * \param StepSize  N-dimensional initial step size vector
   * \param Tol       The user-supplied tolerance
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(std::vector<DataType> const &X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Set the N-dimensional initial vector and initial StepSize
   *
   * \param X         N-dimensional initial vector
   * \param StepSize  N-dimensional initial step size vector
   * \param Tol       The user-supplied tolerance
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(DataType const *X, DataType const StepSize, DataType const Tol);

  /*!
   * \brief Initialize the minimizer
   * 
   * \return false If the iteration encounters an unexpected problem
   */
  virtual bool init();

  /*!
   * \brief Drives the iteration of each algorithm
   *
   * It performs one iteration to update the state of the minimizer.
   *
   * \return false If the iteration encounters an unexpected problem
   */
  virtual bool iterate();

  /*!
   * \brief This function resets the minimizer to use the current point as a new starting point
   *
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
   * \return DataType* N-dimensional x vector
   */
  inline DataType *getX();

  /*!
   * \brief Get N-dimensional dx vector
   *
   * \return DataType* N-dimensional dx vector
   */
  inline DataType *getdX();

  /*!
   * \brief Get N-dimensional gradient vector
   *
   * \return DataType* N-dimensional gradient vector
   */
  inline DataType *getGradient();

  /*!
   * \brief Helper function to test the norm of the gradient against the absolute tolerance, since the gradient goes to zero at a minimum
   * 
   * \param G       Input gradient vector
   * \param abstol  Absolute tolerance
   *  
   * \return -1, 0, and 1 (where -1:Fail, 0:Success, and 1:Continue)
   */
  int testGradient(DataType const *G, DataType const abstol);

  /*!
   * \brief Helper function to test the norm of the gradient against the absolute tolerance, since the gradient goes to zero at a minimum
   * 
   * \param G       Input gradient vector
   * \param abstol  Absolute tolerance
   *  
   * \return -1, 0, and 1 (where -1:Fail, 0:Success, and 1:Continue)
   */
  int testGradient(std::vector<DataType> const &G, DataType const abstol);

  /*!
   * \brief Get the minimum function value
   *
   * \return The minimum function value
   */
  inline DataType getMin();

  /*!
   * \brief Get the number of dimensions 
   * 
   * \returns Number of dimensions
   */
  inline int getDimension();

  /*!
   * \brief Helper function to compute the gradient of the function f at X, \f$~(\frac{\partial f}{\partial x}) \f$
   * 
   * \note 
   * - Helper function to compute the gradient by a finite-difference approximation in one-dimension.
   * - Using this routine is not advised, you should probably use a derivative-free algorithm instead.
   * - Finite-difference approximations are not only expensive, but they are also notoriously susceptible to roundoff 
   * errors. <br>
   * - On the other hand, finite-difference approximations are very useful to check that your analytical 
   * gradient computation is correct—this is always a good idea, because in my experience it is very easy to have 
   * bugs in your gradient code, and an incorrect gradient will cause weird problems with a gradient-based 
   * optimization algorithm.
   * 
   * \param X  Input point
   * \param G  Gradient of the function f at X, \f$~(\frac{\partial f}{\partial x}) \f$
   * 
   * \return false If the iteration encounters an unexpected problem
   */
  bool df(DataType const *X, DataType *G);

  /*!
   * \brief Helper function to compute the function value, and its gradient at X, \f$~(\frac{\partial f}{\partial x})\f$
   * 
   * \param X  Input point
   * \param F  Function value at X
   * \param G  Function gradient \f$ \nabla \f$ at X
   * 
   * \return false If the iteration encounters an unexpected problem
   */
  bool fdf(DataType const *X, DataType *F, DataType *G);

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
  void takeStep(DataType const *X, DataType const *P, DataType const Step, DataType const lambda, DataType *X1, DataType *DX);

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
  void takeStep(std::vector<DataType> const &X, std::vector<DataType> const &P, DataType const Step, DataType const lambda, std::vector<DataType> &X1, std::vector<DataType> &DX);

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
  bool intermediatePoint(DataType const *X, DataType const *P,
                         DataType const lambda, DataType const pg,
                         DataType const stepc,
                         DataType const fa, DataType const fc,
                         DataType *X1, DataType *DX,
                         DataType *Gradient, DataType *Step, DataType *Fval);

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
  bool intermediatePoint(std::vector<DataType> const &X, std::vector<DataType> const &P,
                         DataType const lambda, DataType const pg,
                         DataType const stepc,
                         DataType const fa, DataType const fc,
                         std::vector<DataType> &X1, std::vector<DataType> &DX,
                         std::vector<DataType> &Gradient, DataType &Step, DataType &Fval);

  /*!
   * \brief This function starting at \f$ (x_0, f_0) \f$ move along the direction \f$ p \f$ to find a minimum
   *          \f$ f(x_0 - \lambda * p) \f$, returning the new point \f$ x_1 = x_0-\lambda*p, \f$
   *          \f$ f_1=f(x_1) \f$ and \f$ g_1 = \nabla{f} \f$ at \f$ x_1\f$
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
  bool minimize(DataType const *X, DataType const *P,
                DataType const lambda, DataType const stepa,
                DataType const stepb, DataType const stepc,
                DataType const fa, DataType const fb,
                DataType const fc, DataType const Tol,
                DataType *X1, DataType *DX1,
                DataType *X2, DataType *DX2,
                DataType *Gradient, DataType *Step,
                DataType *Fval, DataType *Gnorm);

  /*!
   * \brief This function starting at \f$ (x_0, f_0) \f$ move along the direction \f$ p \f$ to find a minimum
   *          \f$ f(x_0 - \lambda * p) \f$, returning the new point \f$ x_1 = x_0-\lambda*p, \f$
   *          \f$ f_1=f(x_1) \f$ and \f$ g_1 = \nabla{f} \f$ at \f$ x_1\f$
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
  bool minimize(std::vector<DataType> const &X, std::vector<DataType> const &P,
                DataType const lambda, DataType const stepa,
                DataType const stepb, DataType const stepc,
                DataType const fa, DataType const fb,
                DataType const fc, DataType const Tol,
                std::vector<DataType> &X1, std::vector<DataType> &DX1,
                std::vector<DataType> &X2, std::vector<DataType> &DX2,
                std::vector<DataType> &Gradient, DataType &Step,
                DataType &Fval, DataType &Gnorm);

protected:
  /*!
   * \brief Delete a differentiableFunctionMinimizer object copy construction
   * 
   * Make it noncopyable.
   */
  differentiableFunctionMinimizer(differentiableFunctionMinimizer<DataType> const &) = delete;

  /*!
   * \brief Delete a differentiableFunctionMinimizer object assignment
   * 
   * Make it nonassignable
   * 
   * \returns differentiableFunctionMinimizer<DataType, F>& 
   */
  differentiableFunctionMinimizer<DataType> &operator=(differentiableFunctionMinimizer<DataType> const &) = delete;

public:
  //! Name of the differentiableFunctionMinimizer
  std::string name;

  //! Multi dimensional differentiable function
  umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>> fun;

  //! N-dimensional x vector
  std::vector<DataType> x;

  //! N-dimensional dx vector
  std::vector<DataType> dx;

  //! N-dimensional gradient vector
  std::vector<DataType> gradient;

  //!
  DataType step;

  //!
  DataType maxStep;

  //! Tolerance
  DataType tol;

  //! Function value
  DataType fval;
};

template <typename DataType>
differentiableFunctionMinimizer<DataType>::differentiableFunctionMinimizer(char const *Name) : name(Name) {}

template <typename DataType>
differentiableFunctionMinimizer<DataType>::~differentiableFunctionMinimizer() {}

template <typename DataType>
differentiableFunctionMinimizer<DataType>::differentiableFunctionMinimizer(differentiableFunctionMinimizer<DataType> &&other)
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

template <typename DataType>
differentiableFunctionMinimizer<DataType> &differentiableFunctionMinimizer<DataType>::operator=(differentiableFunctionMinimizer<DataType> &&other)
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

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::reset(int const nDim) noexcept
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

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>> &umFun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>> &umFun, DataType const *X, DataType const StepSize, DataType const Tol)
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, DF_MTYPE<DataType> &DFun, FDF_MTYPE<DataType> &FDFun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, DF_MTYPE<DataType> const &DFun, FDF_MTYPE<DataType> const &FDFun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, DF_MTYPE<DataType> &DFun, FDF_MTYPE<DataType> &FDFun, DataType const *X, DataType const StepSize, DataType const Tol)
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, DF_MTYPE<DataType> const &DFun, FDF_MTYPE<DataType> const &FDFun, DataType const *X, DataType const StepSize, DataType const Tol)
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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

  fun.df = std::bind(&differentiableFunctionMinimizer<DataType>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<DataType>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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

  fun.df = std::bind(&differentiableFunctionMinimizer<DataType>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<DataType>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, DataType const *X, DataType const StepSize, DataType const Tol)
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

  fun.df = std::bind(&differentiableFunctionMinimizer<DataType>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<DataType>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, DataType const *X, DataType const StepSize, DataType const Tol)
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

  fun.df = std::bind(&differentiableFunctionMinimizer<DataType>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<DataType>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, DF_MTYPE<DataType> &DFun, FDF_MTYPE<DataType> &FDFun, std::vector<DataType> const &Params, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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
    fun = std::move(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>>(Params));
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, std::vector<DataType> const &Params, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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
    fun = std::move(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  fun.df = std::bind(&differentiableFunctionMinimizer<DataType>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<DataType>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, DF_MTYPE<DataType> const &DFun, FDF_MTYPE<DataType> const &FDFun, std::vector<DataType> const &Params, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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
    fun = std::move(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>>(Params));
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, std::vector<DataType> const &Params, std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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
    fun = std::move(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  fun.df = std::bind(&differentiableFunctionMinimizer<DataType>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<DataType>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, DF_MTYPE<DataType> &DFun, FDF_MTYPE<DataType> &FDFun, DataType const *Params, int const NumParams, DataType const *X, DataType const StepSize, DataType const Tol)
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
    fun = std::move(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>>(Params, NumParams));
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, DataType const *Params, int const NumParams, DataType const *X, DataType const StepSize, DataType const Tol)
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
    fun = std::move(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  fun.df = std::bind(&differentiableFunctionMinimizer<DataType>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<DataType>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, DF_MTYPE<DataType> const &DFun, FDF_MTYPE<DataType> const &FDFun, DataType const *Params, int const NumParams, DataType const *X, DataType const StepSize, DataType const Tol)
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
    fun = std::move(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>>(Params, NumParams));
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, DataType const *Params, int const NumParams, DataType const *X, DataType const StepSize, DataType const Tol)
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
    fun = std::move(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  fun.df = std::bind(&differentiableFunctionMinimizer<DataType>::df, this, std::placeholders::_1, std::placeholders::_2);
  fun.fdf = std::bind(&differentiableFunctionMinimizer<DataType>::fdf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  // Set dx to zero
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(std::vector<DataType> const &X, DataType const StepSize, DataType const Tol)
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::set(DataType const *X, DataType const StepSize, DataType const Tol)
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
  std::fill(dx.begin(), dx.end(), DataType{});

  step = StepSize;
  maxStep = StepSize;
  tol = Tol;

  return fun.fdf(x.data(), &fval, gradient.data());
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::init()
{
  return true;
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::iterate()
{
  return true;
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::restart()
{
  return true;
}

template <typename DataType>
inline std::string const differentiableFunctionMinimizer<DataType>::getName() const
{
  return name;
}

template <typename DataType>
inline DataType *differentiableFunctionMinimizer<DataType>::getX()
{
  return x.data();
}

template <typename DataType>
inline DataType *differentiableFunctionMinimizer<DataType>::getdX()
{
  return dx.data();
}

template <typename DataType>
inline DataType *differentiableFunctionMinimizer<DataType>::getGradient()
{
  return gradient.data();
}

template <typename DataType>
inline int differentiableFunctionMinimizer<DataType>::testGradient(DataType const *G, DataType const abstol)
{
  if (abstol < DataType{})
  {
    UMUQWARNING("Absolute tolerance is negative!");
    // fail
    return -1;
  }

  int const n = getDimension();

  // Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
  DataType norm(0);
  std::for_each(G, G + n, [&](DataType const g_i) { norm += g_i * g_i; });

  return std::sqrt(norm) >= abstol;
}

template <typename DataType>
inline int differentiableFunctionMinimizer<DataType>::testGradient(std::vector<DataType> const &G, DataType const abstol)
{
  if (abstol < DataType{})
  {
    UMUQWARNING("Absolute tolerance is negative!");
    // fail
    return -1;
  }

  // Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
  DataType norm(0);
  std::for_each(G.begin(), G.end(), [&](DataType const g_i) { norm += g_i * g_i; });

  return std::sqrt(norm) >= abstol;
}

template <typename DataType>
inline DataType differentiableFunctionMinimizer<DataType>::getMin()
{
  return fval;
}

template <typename DataType>
inline int differentiableFunctionMinimizer<DataType>::getDimension()
{
  return x.size();
}

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::df(DataType const *X, DataType *G)
{
  if (!G)
  {
    UMUQFAILRETURN("The gradient pointer is not assigned!");
  }

  DataType const h = std::sqrt(std::numeric_limits<DataType>::epsilon());

  // Get the number of dimensions
  int const n = getDimension();

  std::vector<DataType> X1(X, X + n);

  for (int i = 0; i < n; i++)
  {
    DataType low;
    DataType high;

    DataType XI = X[i];

    DataType DX = std::abs(XI) * h;
    if (DX <= DataType{})
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

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::fdf(DataType const *X, DataType *F, DataType *G)
{
  if (fun)
  {
    *F = fun.f(X);
    return df(X, G);
  }
  UMUQFAILRETURN("The function is not assigned!");
}

template <typename DataType>
void differentiableFunctionMinimizer<DataType>::takeStep(DataType const *X, DataType const *P, DataType const Step, DataType const lambda, DataType *X1, DataType *DX)
{
  int const n = getDimension();

  // Compute the sum \f$y = \alpha x + y\f$ for the vectors x and y (set dx to zero).
  DataType const alpha = -Step * lambda;

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

template <typename DataType>
void differentiableFunctionMinimizer<DataType>::takeStep(std::vector<DataType> const &X, std::vector<DataType> const &P, DataType const Step, DataType const lambda, std::vector<DataType> &X1, std::vector<DataType> &DX)
{
  int const n = getDimension();

  // Compute the sum \f$y = \alpha x + y\f$ for the vectors x and y (set dx to zero).
  DataType const alpha = -Step * lambda;

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

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::intermediatePoint(DataType const *X, DataType const *P,
                                                                  DataType const lambda, DataType const pg,
                                                                  DataType const stepc,
                                                                  DataType const fa, DataType const fc,
                                                                  DataType *X1, DataType *DX,
                                                                  DataType *Gradient, DataType *Step, DataType *Fval)
{
  DataType stepb(1);
  DataType stepd(stepc);

  DataType fb(fa);
  DataType fd(fc);

  while (fb >= fa && stepb > DataType{})
  {
    DataType u = std::abs(pg * lambda * stepd);

    stepb = 0.5 * stepd * u / ((fd - fa) + u);

    takeStep(X, P, stepb, lambda, X1, DX);

    fb = fun.f(X1);

    if (fb >= fa && stepb > DataType{})
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

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::intermediatePoint(std::vector<DataType> const &X, std::vector<DataType> const &P,
                                                                  DataType const lambda, DataType const pg,
                                                                  DataType const stepc,
                                                                  DataType const fa, DataType const fc,
                                                                  std::vector<DataType> &X1, std::vector<DataType> &DX,
                                                                  std::vector<DataType> &Gradient, DataType &Step, DataType &Fval)
{
  DataType stepb(1);
  DataType stepd(stepc);

  DataType fb(fa);
  DataType fd(fc);

  while (fb >= fa && stepb > DataType{})
  {
    DataType u = std::abs(pg * lambda * stepd);

    stepb = 0.5 * stepd * u / ((fd - fa) + u);

    takeStep(X, P, stepb, lambda, X1, DX);

    fb = fun.f(X1.data());

    if (fb >= fa && stepb > DataType{})
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

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::minimize(DataType const *X, DataType const *P,
                                                         DataType const lambda, DataType const stepa,
                                                         DataType const stepb, DataType const stepc,
                                                         DataType const fa, DataType const fb,
                                                         DataType const fc, DataType const Tol,
                                                         DataType *X1, DataType *DX1,
                                                         DataType *X2, DataType *DX2,
                                                         DataType *Gradient, DataType *Step,
                                                         DataType *Fval, DataType *Gnorm)
{
  // Starting at (x0, f0) move along the direction p to find a minimum
  // \f$ f(x0 - lambda * p) \f$, returning the new point \f$ x1 = x0-lambda*p, \f$
  // \f$ f1=f(x1) \f$ and \f$ g1 = grad(f) \f$ at x1

  int const n = getDimension();

  DataType stpb(stepb);
  DataType stpa(stepa);
  DataType stpc(stepc);

  DataType fstpb(fb);

  DataType u(stepb);
  DataType v(stepa);
  DataType w(stepc);

  DataType fu(fb);
  DataType fv(fa);
  DataType fw(fc);

  DataType old2 = std::abs(w - v);
  DataType old1 = std::abs(v - u);

  DataType stepm;
  DataType fm;
  DataType pg;
  DataType gnorm1;

  std::copy(X1, X1 + n, X2);
  std::copy(DX1, DX1 + n, DX2);

  *Fval = fstpb;
  *Step = stpb;

  DataType s(0);
  std::for_each(Gradient, Gradient + n, [&](DataType const g_i) { s += g_i * g_i; });
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
      DataType dw = w - u;
      DataType dv = v - u;

      DataType e1 = ((fv - fu) * dw * dw + (fu - fw) * dv * dv);
      DataType e2 = 2 * ((fv - fu) * dw + (fu - fw) * dv);

      DataType du(0);
      if (e2 != DataType{})
      {
        du = e1 / e2;
      }

      if (du > DataType{} && du < (stpc - stpb) && std::abs(du) < 0.5 * old2)
      {
        stepm = u + du;
      }
      else if (du < DataType{} && du > (stpa - stpb) && std::abs(du) < 0.5 * old2)
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

      pg = DataType{};
      for (int i = 0; i < n; i++)
      {
        pg += P[i] * Gradient[i];
      }

      s = (DataType)0;
      std::for_each(Gradient, Gradient + n, [&](DataType const g_i) { s += g_i * g_i; });
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

template <typename DataType>
bool differentiableFunctionMinimizer<DataType>::minimize(std::vector<DataType> const &X, std::vector<DataType> const &P,
                                                         DataType const lambda, DataType const stepa,
                                                         DataType const stepb, DataType const stepc,
                                                         DataType const fa, DataType const fb,
                                                         DataType const fc, DataType const Tol,
                                                         std::vector<DataType> &X1, std::vector<DataType> &DX1,
                                                         std::vector<DataType> &X2, std::vector<DataType> &DX2,
                                                         std::vector<DataType> &Gradient, DataType &Step,
                                                         DataType &Fval, DataType &Gnorm)
{
  // Starting at (x0, f0) move along the direction p to find a minimum
  // \f$ f(x0 - lambda * p) \f$, returning the new point \f$ x1 = x0-lambda*p, \f$
  // \f$ f1=f(x1) \f$ and \f$ g1 = grad(f) \f$ at x1

  int const n = getDimension();

  DataType stpb(stepb);
  DataType stpa(stepa);
  DataType stpc(stepc);

  DataType fstpb(fb);

  DataType u(stepb);
  DataType v(stepa);
  DataType w(stepc);

  DataType fu(fb);
  DataType fv(fa);
  DataType fw(fc);

  DataType old2 = std::abs(w - v);
  DataType old1 = std::abs(v - u);

  DataType stepm;
  DataType fm;
  DataType pg;
  DataType gnorm1;

  std::copy(X1.begin(), X1.end(), X2.begin());
  std::copy(DX1.begin(), DX1.end(), DX2.begin());

  Fval = fstpb;
  Step = stpb;

  DataType s(0);
  std::for_each(Gradient.begin(), Gradient.end(), [&](DataType const g_i) { s += g_i * g_i; });
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
      DataType dw = w - u;
      DataType dv = v - u;

      DataType e1 = ((fv - fu) * dw * dw + (fu - fw) * dv * dv);
      DataType e2 = 2 * ((fv - fu) * dw + (fu - fw) * dv);

      DataType du(0);
      if (e2 != DataType{})
      {
        du = e1 / e2;
      }

      if (du > DataType{} && du < (stpc - stpb) && std::abs(du) < 0.5 * old2)
      {
        stepm = u + du;
      }
      else if (du < DataType{} && du > (stpa - stpb) && std::abs(du) < 0.5 * old2)
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

      pg = DataType{};
      for (int i = 0; i < n; i++)
      {
        pg += P[i] * Gradient[i];
      }

      s = (DataType)0;
      std::for_each(Gradient.begin(), Gradient.end(), [&](DataType const g_i) { s += g_i * g_i; });
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
