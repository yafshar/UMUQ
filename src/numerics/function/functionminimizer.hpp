#ifndef UMUQ_FUNCTIONMINIMIZER_H
#define UMUQ_FUNCTIONMINIMIZER_H

#include "functiontype.hpp"
#include "umuqfunction.hpp"

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
 */
template <typename T>
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
  virtual bool set(umuqFunction<T, F_MTYPE<T>> &umFun, std::vector<T> const &X, std::vector<T> const &stepSize);
  virtual bool set(umuqFunction<T, F_MTYPE<T>> &umFun, T const *X, T const *stepSize);

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
  virtual bool set(F_MTYPE<T> &Fun, std::vector<T> const &X, std::vector<T> const &stepSize);
  virtual bool set(F_MTYPE<T> const &Fun, std::vector<T> const &X, std::vector<T> const &stepSize);

  virtual bool set(F_MTYPE<T> &Fun, T const *X, T const *stepSize);
  virtual bool set(F_MTYPE<T> const &Fun, T const *X, T const *stepSize);

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
  virtual bool set(F_MTYPE<T> &Fun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize);
  virtual bool set(F_MTYPE<T> const &Fun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize);

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
  virtual bool set(F_MTYPE<T> &Fun, T const *Params, int const NumParams, T const *X, T const *stepSize);
  virtual bool set(F_MTYPE<T> const &Fun, T const *Params, int const NumParams, T const *X, T const *stepSize);

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
   * \brief Initilize the minimizer
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
   * \brief Get the Name object
   * 
   * \return std::string const 
   */
  inline std::string const getName() const;

  /*!
   * \brief Return minimizer-specific characteristic size
   * 
   * \returns minimizer-specific characteristic size
   */
  inline T const getSize() const;

  /*!
   * \brief Helper function to check the specific characteristic size against absolute tolerance
   * 
   * \param abstol Absolute tolerance
   * 
   * \return -1, 0, and 1 (where -1:Fail, 0:Success, and 1:Continue) 
   */
  inline int testSize(T const abstol);

  /*!
   * \brief Get the N-dimensional x vector
   * 
   * \return T* 
   */
  inline T *getX();

  /*!
   * \brief Get the minimum function value
   * 
   * \return the minimum function value
   */
  inline T getMin();

  /*!
   * \brief Get the number of Dimensions 
   * 
   * \return the number of dimensions
   */
  inline int getDimension();

private:
  // Make it noncopyable
  functionMinimizer(functionMinimizer<T> const &) = delete;

  // Make it not assignable
  functionMinimizer<T> &operator=(functionMinimizer<T> const &) = delete;

public:
  //! Name of the functionMinimizer
  std::string name;

  //! Function to be used in this minimizer
  umuqFunction<T, F_MTYPE<T>> fun;

  //! N-dimensional x vector
  std::vector<T> x;

  // Workspace 1 for algorithm
  std::vector<T> ws1;

  // Workspace 2 for algorithm
  std::vector<T> ws2;

  //! The minimizer-specific characteristic size (This size can be used as a stopping criteria)
  T size;

  //! Minimum function value
  T fval;
};

template <typename T>
functionMinimizer<T>::functionMinimizer(char const *Name) : name(Name) {}

template <typename T>
bool functionMinimizer<T>::reset(int const nDim) noexcept
{
  if (nDim <= 0)
  {
    UMUQFAILRETURN("Invalid number of parameters specified!");
  }

  x.resize(nDim);
  ws1.resize(nDim);
  ws2.resize(nDim);

  return true;
}

template <typename T>
bool functionMinimizer<T>::set(umuqFunction<T, F_MTYPE<T>> &umFun, std::vector<T> const &X, std::vector<T> const &stepSize)
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
  std::copy(stepSize.begin(), stepSize.end(), ws2.begin());

  return true;
}

template <typename T>
bool functionMinimizer<T>::set(umuqFunction<T, F_MTYPE<T>> &umFun, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
    std::copy(stepSize, stepSize + x.size(), ws2.begin());
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

template <typename T>
bool functionMinimizer<T>::set(F_MTYPE<T> &Fun, std::vector<T> const &X, std::vector<T> const &stepSize)
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
  std::copy(stepSize.begin(), stepSize.end(), ws2.begin());

  return true;
}

template <typename T>
bool functionMinimizer<T>::set(F_MTYPE<T> const &Fun, std::vector<T> const &X, std::vector<T> const &stepSize)
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
  std::copy(stepSize.begin(), stepSize.end(), ws2.begin());

  return true;
}

template <typename T>
bool functionMinimizer<T>::set(F_MTYPE<T> &Fun, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
    std::copy(stepSize, stepSize + x.size(), ws2.begin());
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

template <typename T>
bool functionMinimizer<T>::set(F_MTYPE<T> const &Fun, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
    std::copy(stepSize, stepSize + x.size(), ws2.begin());
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

template <typename T>
bool functionMinimizer<T>::set(F_MTYPE<T> &Fun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize)
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
    fun = std::move(umuqFunction<T, F_MTYPE<T>>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X.begin(), X.end(), x.begin());
  std::copy(stepSize.begin(), stepSize.end(), ws2.begin());

  return true;
}

template <typename T>
bool functionMinimizer<T>::set(F_MTYPE<T> const &Fun, std::vector<T> const &Params, std::vector<T> const &X, std::vector<T> const &stepSize)
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
    fun = std::move(umuqFunction<T, F_MTYPE<T>>(Params));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X.begin(), X.end(), x.begin());
  std::copy(stepSize.begin(), stepSize.end(), ws2.begin());

  return true;
}

template <typename T>
bool functionMinimizer<T>::set(F_MTYPE<T> &Fun, T const *Params, int const NumParams, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
    std::copy(stepSize, stepSize + x.size(), ws2.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqFunction<T, F_MTYPE<T>>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  return true;
}

template <typename T>
bool functionMinimizer<T>::set(F_MTYPE<T> const &Fun, T const *Params, int const NumParams, T const *X, T const *stepSize)
{
  if (x.size() > 0)
  {
    std::copy(X, X + x.size(), x.begin());
    std::copy(stepSize, stepSize + x.size(), ws2.begin());
  }
  else
  {
    UMUQFAILRETURN("Incompatible input vector size with solver size!");
  }

  if (Fun)
  {
    fun = std::move(umuqFunction<T, F_MTYPE<T>>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  return true;
}

template <typename T>
bool functionMinimizer<T>::set(std::vector<T> const &X, std::vector<T> const &stepSize)
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
  std::copy(stepSize.begin(), stepSize.end(), ws2.begin());

  return true;
}

template <typename T>
bool functionMinimizer<T>::set(T const *X, T const *stepSize)
{
  if (!fun)
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X, X + x.size(), x.begin());
  std::copy(stepSize, stepSize + x.size(), ws2.begin());

  return true;
}

template <typename T>
bool functionMinimizer<T>::init()
{
  return true;
}

template <typename T>
bool functionMinimizer<T>::iterate()
{
  return true;
}

template <typename T>
inline std::string const functionMinimizer<T>::getName() const
{
  return name;
}

template <typename T>
inline T const functionMinimizer<T>::getSize() const
{
  return size;
}

template <typename T>
inline int functionMinimizer<T>::testSize(T const abstol)
{
  return (abstol < 0) ? -1 : ((size < abstol) ? 0 : 1);
}

template <typename T>
inline T *functionMinimizer<T>::getX()
{
  return x.data();
}

template <typename T>
inline T functionMinimizer<T>::getMin()
{
  return fval;
}

template <typename T>
inline int functionMinimizer<T>::getDimension()
{
  return x.size();
}

/*!
 * \brief A helper function for testing the function minimizer 
 * 
 * \tparam T          Data type
 * \param fMinimizer  Function Minimizer object
 * \param Fun         Function to be minimized 
 * \param X           N-Dimensional input data
 * \param nDim        Dimension of the data
 * \param FunName     Function name or description
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool functionMinimizerTest(functionMinimizer<T> &fMinimizer, F_MTYPE<T> const &Fun, T const *X, int const nDim, char const *FunName)
{
  //! By default we consider stepSize 1
  std::vector<T> stepSize(nDim, 1);

  //! First we have to set the minimizer dimension
  if (!fMinimizer.reset(nDim))
  {
    UMUQFAILRETURN("Failed to set the minimizer dimension!");
  }

  //! Second, we have to set the function, input vector and stepsize
  if (!fMinimizer.set(Fun, X, stepSize.data()))
  {
    UMUQFAILRETURN("Failed to set the minimizer!");
  }

  //! Third, initilize the minimizer
  if (!fMinimizer.init())
  {
    UMUQFAILRETURN("Failed to initialize the minimizer!");
  }

#ifdef DEBUG
  {
    T *x = fMinimizer.getX();

    std::cout << "x =";
    for (int i = 0; i < nDim; i++)
    {
      std::cout << x[i] << " ";
    }
    std::cout << std::endl;
  }
#endif

  // Forth, iterate until we reach the absolute tolerance of 1e-3

  // Fail:-1, Success:0, Continue:1
  int status = 1;
  int iter = 0;

  while (iter < 5000 && status == 1)
  {
    iter++;

    if (!fMinimizer.iterate())
    {
      UMUQFAILRETURN("Failed to iterate the minimizer!");
    }

#ifdef DEBUG
    {
      std::cout << iter << ": ";

      T *x = fMinimizer.getX();

      std::cout << "x = ";
      for (int i = 0; i < nDim; i++)
      {
        std::cout << x[i] << " ";
      }
      std::cout << std::endl;

      std::cout << "f(x) =" << fMinimizer.getMin() << ", & characteristic size =" << fMinimizer.getSize() << std::endl;
    }
#endif

    status = fMinimizer.testSize(1e-3);
  }

  if (status == 0 || status == 1)
  {
    std::cout << fMinimizer.getName() << ", on " << FunName << ": " << iter << " iters, f(x)=" << fMinimizer.getMin() << std::endl;
    std::cout << (status == 0) ? "Converged to minimum at x = " : "Stopped at x = ";

    T *x = fMinimizer.getX();
    for (int i = 0; i < nDim; i++)
    {
      std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    return status ? (std::abs(fMinimizer.getMin()) > 1e-5) : true;
  }
  return false;
}

#endif //UMUQ_FUNCTIONMINIMIZER
