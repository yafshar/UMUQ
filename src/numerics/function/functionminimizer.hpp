#ifndef UMUQ_FUNCTIONMINIMIZER_H
#define UMUQ_FUNCTIONMINIMIZER_H

#include "datatype/functiontype.hpp"
#include "datatype/differentiablefunctionminimizertype.hpp"
#include "umuqfunction.hpp"

namespace umuq
{

/*! 
 * \defgroup Multimin_Module Multimin module
 * \ingroup Numerics_Module
 *
 * This is the Multidimensional Minimization Module of UMUQ providing all necessary classes 
 * for finding minima of arbitrary multidimensional functions.
 */

/*! 
 * \namespace umuq::multimin
 * \ingroup Multimin_Module
 * 
 * \brief Namespace containing all the functions for Multidimensional Minimization Module
 * 
 * It includes all the functionalities for finding minima of arbitrary multidimensional 
 * functions. It provides low level components for a variety of iterative minimizers 
 * and convergence tests. 
 */
inline namespace multimin
{
/*! \class functionMinimizer
 * \ingroup Multimin_Module
 * 
 * \brief The goal is finding minima of arbitrary multidimensional functions.<br> 
 * This is the base class which is for finding minima of arbitrary multidimensional functions.
 * 
 * \tparam DataType Data type
 * 
 * 
 * This is the low level component for a variety of iterative minimizers and the base class 
 * suitable for algorithms which do not require the gradient of the function. <br>
 * For example, the Nelder-Mead Simplex algorithm.
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
 */
template <typename DataType>
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
   * \brief Destroy the function Minimizer object
   * 
   */
  ~functionMinimizer();

  /*!
   * \brief Move constructor, Construct a new functionMinimizer object
   * 
   * \param other functionMinimizer object
   */
  functionMinimizer(functionMinimizer<DataType> &&other);

  /*!
   * \brief Move assignment operator
   * 
   */
  functionMinimizer<DataType> &operator=(functionMinimizer<DataType> &&other);

  /*!
   * \brief Resizes the x-vector to contain nDim elements 
   * 
   * \param nDim  New size of the x-vector
   * 
   * \returns true  
   */
  virtual bool reset(int const nDim) noexcept;

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param umFun     umuq Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(umuqFunction<DataType, F_MTYPE<DataType>> &umFun, std::vector<DataType> const &X, std::vector<DataType> const &stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param umFun     umuq Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(umuqFunction<DataType, F_MTYPE<DataType>> &umFun, DataType const *X, DataType const *stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param Fun       Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, std::vector<DataType> const &X, std::vector<DataType> const &stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param Fun       Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, std::vector<DataType> const &X, std::vector<DataType> const &stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param Fun       Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, DataType const *X, DataType const *stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   * 
   * \param Fun       Function to be used in this minimizer
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   * 
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, DataType const *X, DataType const *stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   *
   * \param Fun       Function to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, std::vector<DataType> const &Params, std::vector<DataType> const &X, std::vector<DataType> const &stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   *
   * \param Fun       Function to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, std::vector<DataType> const &Params, std::vector<DataType> const &X, std::vector<DataType> const &stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   *
   * \param Fun       Function to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param NumParams Number of dimensions (Number of parameters of the Function object)
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> &Fun, DataType const *Params, int const NumParams, DataType const *X, DataType const *stepSize);

  /*!
   * \brief Set the Function to be used in this minimizer, N-dimensional initial vector and initial stepSize
   *
   * \param Fun       Function to be used in this minimizer
   * \param Params    Input parameters of the Function object
   * \param NumParams Number of dimensions (Number of parameters of the Function object)
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(F_MTYPE<DataType> const &Fun, DataType const *Params, int const NumParams, DataType const *X, DataType const *stepSize);

  /*!
   * \brief Set the N-dimensional initial vector and initial stepSize
   *
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(std::vector<DataType> const &X, std::vector<DataType> const &stepSize);

  /*!
   * \brief Set the N-dimensional initial vector and initial stepSize
   *
   * \param X         N-dimensional initial vector
   * \param stepSize  N-dimensional initial step size vector
   *
   * \return false If it encounters an unexpected problem
   */
  virtual bool set(DataType const *X, DataType const *stepSize);

  /*!
   * \brief Initialize the minimizer
   * 
   * \return false If it encounters an unexpected problem
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
  inline DataType const size() const;

  /*!
   * \brief Helper function to check the specific characteristic size against absolute tolerance
   * 
   * \param abstol Absolute tolerance
   * 
   * \return -1, 0, and 1 (where -1:Fail, 0:Success, and 1:Continue) 
   */
  inline int testSize(DataType const abstol);

  /*!
   * \brief Get the N-dimensional x vector
   * 
   * \return DataType* Get the N-dimensional x vector
   */
  inline DataType *getX();

  /*!
   * \brief Get the minimum function value
   * 
   * \return the minimum function value
   */
  inline DataType getMin();

  /*!
   * \brief Get the number of Dimensions 
   * 
   * \return the number of dimensions
   */
  inline int getDimension();

protected:
  /*!
   * \brief Delete a functionMinimizer object copy construction
   * 
   * Avoiding implicit generation of the copy constructor.
   */
  functionMinimizer(functionMinimizer<DataType> const &) = delete;

  /*!
   * \brief Delete a functionMinimizer object assignment
   * 
   * Avoiding implicit copy assignment.
   * 
   * \returns functionMinimizer<DataType>& 
   */
  functionMinimizer<DataType> &operator=(functionMinimizer<DataType> const &) = delete;

public:
  //! Name of the functionMinimizer
  std::string name;

  //! Function to be used in this minimizer
  umuqFunction<DataType, F_MTYPE<DataType>> fun;

  //! N-dimensional x vector
  std::vector<DataType> x;

  //! Workspace 1 for the algorithm
  std::vector<DataType> ws1;

  //! Workspace 2 for the algorithm
  std::vector<DataType> ws2;

  //! The minimizer-specific characteristic size (This size can be used as a stopping criteria)
  DataType characteristicSize;

  //! Minimum function value
  DataType fval;
};

template <typename DataType>
functionMinimizer<DataType>::functionMinimizer(char const *Name) : name(Name) {}

template <typename DataType>
functionMinimizer<DataType>::~functionMinimizer() {}

template <typename DataType>
functionMinimizer<DataType>::functionMinimizer(functionMinimizer<DataType> &&other)
{
  name = other.name;
  fun = std::move(other.fun);
  x = std::move(other.x);
  ws1 = std::move(other.ws1);
  ws2 = std::move(other.ws2);
  characteristicSize = other.characteristicSize;
  fval = other.fval;
}

template <typename DataType>
functionMinimizer<DataType> &functionMinimizer<DataType>::operator=(functionMinimizer<DataType> &&other)
{
  name = other.name;
  fun = std::move(other.fun);
  x = std::move(other.x);
  ws1 = std::move(other.ws1);
  ws2 = std::move(other.ws2);
  characteristicSize = other.characteristicSize;
  fval = other.fval;

  return *this;
}

template <typename DataType>
bool functionMinimizer<DataType>::reset(int const nDim) noexcept
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

template <typename DataType>
bool functionMinimizer<DataType>::set(umuqFunction<DataType, F_MTYPE<DataType>> &umFun, std::vector<DataType> const &X, std::vector<DataType> const &stepSize)
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

template <typename DataType>
bool functionMinimizer<DataType>::set(umuqFunction<DataType, F_MTYPE<DataType>> &umFun, DataType const *X, DataType const *stepSize)
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

template <typename DataType>
bool functionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, std::vector<DataType> const &X, std::vector<DataType> const &stepSize)
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

template <typename DataType>
bool functionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, std::vector<DataType> const &X, std::vector<DataType> const &stepSize)
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

template <typename DataType>
bool functionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, DataType const *X, DataType const *stepSize)
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

template <typename DataType>
bool functionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, DataType const *X, DataType const *stepSize)
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

template <typename DataType>
bool functionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, std::vector<DataType> const &Params, std::vector<DataType> const &X, std::vector<DataType> const &stepSize)
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
    fun = std::move(umuqFunction<DataType, F_MTYPE<DataType>>(Params));
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

template <typename DataType>
bool functionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, std::vector<DataType> const &Params, std::vector<DataType> const &X, std::vector<DataType> const &stepSize)
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
    fun = std::move(umuqFunction<DataType, F_MTYPE<DataType>>(Params));
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

template <typename DataType>
bool functionMinimizer<DataType>::set(F_MTYPE<DataType> &Fun, DataType const *Params, int const NumParams, DataType const *X, DataType const *stepSize)
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
    fun = std::move(umuqFunction<DataType, F_MTYPE<DataType>>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  return true;
}

template <typename DataType>
bool functionMinimizer<DataType>::set(F_MTYPE<DataType> const &Fun, DataType const *Params, int const NumParams, DataType const *X, DataType const *stepSize)
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
    fun = std::move(umuqFunction<DataType, F_MTYPE<DataType>>(Params, NumParams));
    fun.f = Fun;
  }
  else
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  return true;
}

template <typename DataType>
bool functionMinimizer<DataType>::set(std::vector<DataType> const &X, std::vector<DataType> const &stepSize)
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

template <typename DataType>
bool functionMinimizer<DataType>::set(DataType const *X, DataType const *stepSize)
{
  if (!fun)
  {
    UMUQFAILRETURN("Function is not assigned!");
  }

  std::copy(X, X + x.size(), x.begin());
  std::copy(stepSize, stepSize + x.size(), ws2.begin());

  return true;
}

template <typename DataType>
bool functionMinimizer<DataType>::init()
{
  return true;
}

template <typename DataType>
bool functionMinimizer<DataType>::iterate()
{
  return true;
}

template <typename DataType>
inline std::string const functionMinimizer<DataType>::getName() const
{
  return name;
}

template <typename DataType>
inline DataType const functionMinimizer<DataType>::size() const
{
  return characteristicSize;
}

template <typename DataType>
inline int functionMinimizer<DataType>::testSize(DataType const abstol)
{
  return (abstol < 0) ? -1 : ((characteristicSize < abstol) ? 0 : 1);
}

template <typename DataType>
inline DataType *functionMinimizer<DataType>::getX()
{
  return x.data();
}

template <typename DataType>
inline DataType functionMinimizer<DataType>::getMin()
{
  return fval;
}

template <typename DataType>
inline int functionMinimizer<DataType>::getDimension()
{
  return x.size();
}

} // namespace multimin
} // namespace umuq

#endif //UMUQ_FUNCTIONMINIMIZER
