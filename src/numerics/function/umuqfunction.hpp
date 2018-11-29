#ifndef UMUQ_UMUQFUNCTION_H
#define UMUQ_UMUQFUNCTION_H

namespace umuq
{

/*!\class umuqFunction
 * \brief umuqFunction is a general-purpose polymorphic function wrapper of n variables
 *
 * \tparam DataType     Data type
 * \tparam FunctionType Function type (wrapped as std::function)
 */
template <typename DataType, class FunctionType>
class umuqFunction
{
public:
  /*!
   * \brief Construct a new umuqFunction object
   * 
   * \param Name  Function name
   */
  explicit umuqFunction(char const *Name = "");

  /*!
   * \brief Construct a new umuqFunction object
   * 
   * \param nDim  Number of dimensions (Number of parameters) 
   * \param Name  Function name
   */
  umuqFunction(int const nDim, char const *Name = "");

  /*!
   * \brief Construct a new umuqFunction object
   * 
   * \param Params    Input parameters of the Function object
   * \param NumParams Number of dimensions (Number of parameters) 
   * \param Name      Function name
   */
  umuqFunction(DataType const *Params, int const NumParams, char const *Name = "");

  /*!
   * \brief Construct a new umuqFunction object
   * 
   * \param Params1    Input parameters of the Function object
   * \param Params2    Input parameters of the Function object
   * \param NumParams  Number of dimensions (Number of parameters) 
   * \param Name       Function name
   */
  umuqFunction(DataType const *Params1, DataType const *Params2, int const NumParams, char const *Name = "");

  /*!
   * \brief Construct a new umuqFunction object
   *  
   * \param Params  Input parameters of the Function object
   * \param Name    Function name
   */
  umuqFunction(std::vector<DataType> const &Params, char const *Name = "");

  /*!
   * \brief Construct a new umuqFunction object
   *  
   * \param Params1  Input parameters of the Function object
   * \param Params2  Input parameters of the Function object
   * \param Name    Function name
   */
  umuqFunction(std::vector<DataType> const &Params1, std::vector<DataType> const &Params2, char const *Name = "");

  /*!
   * \brief Destroy the umuq Function object
   * 
   */
  ~umuqFunction();

  /*!
   * \brief Move constructor, Construct a new umuqFunction object
   * 
   * \param other umuqFunction object
   */
  umuqFunction(umuqFunction<DataType, FunctionType> &&other);

  /*!
   * \brief Move assignment operator
   * 
   */
  umuqFunction<DataType, FunctionType> &operator=(umuqFunction<DataType, FunctionType> &&other);

  /*!
   * \brief Get the Name object
   * 
   * \return std::string const 
   */
  std::string const getName() const;

  /*!
   * \brief Set the Name object
   * 
   */
  void setName(char const *Name);

  /*!
   * \brief Checks whether *this stores a callable function target, i.e. is not empty. 
   * 
   * \return true If it stores a callable function target at f
   */
  explicit operator bool() const noexcept;

protected:
  /*!
   * \brief Delete a umuqFunction object copy construction
   * 
   * Avoiding implicit generation of the copy constructor.
   */
  umuqFunction(umuqFunction<DataType, FunctionType> const &) = delete;

  /*!
   * \brief Delete a umuqFunction object assignment
   * 
   * Avoiding implicit copy assignment.
   * 
   * \returns umuqFunction<DataType, FunctionType>& 
   */
  umuqFunction<DataType, FunctionType> &operator=(umuqFunction<DataType, FunctionType> const &) = delete;

public:
  //! Name of the function
  std::string name;

  //! Number of dimensions
  std::size_t numParams;

  //! Function parameters
  std::vector<DataType> params;

public:
  /*!
   * \brief A general-purpose polymorphic function wrapper 
   * 
   */
  FunctionType f;
};

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType>::umuqFunction(char const *Name) : name(Name),
                                                                       numParams(0),
                                                                       f(nullptr)
{
}

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType>::umuqFunction(int const nDim, char const *Name) : name(Name),
                                                                                       numParams(nDim > 0 ? nDim : 0),
                                                                                       f(nullptr)
{
}

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType>::umuqFunction(DataType const *Params, int const NumParams, char const *Name) : name(Name),
                                                                                                                    numParams(NumParams > 0 ? NumParams : 0),
                                                                                                                    params(Params, Params + NumParams),
                                                                                                                    f(nullptr)
{
}

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType>::umuqFunction(DataType const *Params1, DataType const *Params2, int const NumParams, char const *Name) : name(Name),
                                                                                                                                              numParams(NumParams > 0 ? NumParams : 0),
                                                                                                                                              f(nullptr)
{
  if (numParams & 1)
  {
    UMUQFAIL("Wrong input size!");
  }
  params.resize(numParams);
  for (std::size_t i = 0, k = 0; i < numParams / 2; i++)
  {
    params[k++] = Params1[i];
    params[k++] = Params2[i];
  }
}

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType>::umuqFunction(std::vector<DataType> const &Params, char const *Name) : name(Name),
                                                                                                            numParams(Params.size()),
                                                                                                            params(Params),
                                                                                                            f(nullptr)
{
}

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType>::umuqFunction(std::vector<DataType> const &Params1, std::vector<DataType> const &Params2, char const *Name) : name(Name),
                                                                                                                                                   numParams(Params1.size() + Params2.size()),
                                                                                                                                                   f(nullptr)
{
  if (Params1.size() != Params2.size())
  {
    UMUQFAIL("Wrong input size!");
  }
  params.resize(numParams);
  for (std::size_t i = 0, k = 0; i < numParams / 2; i++)
  {
    params[k++] = Params1[i];
    params[k++] = Params2[i];
  }
}

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType>::~umuqFunction() {}

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType>::umuqFunction(umuqFunction<DataType, FunctionType> &&other) : name(other.name),
                                                                                                   numParams(other.numParams),
                                                                                                   params(std::move(other.params)),
                                                                                                   f(std::move(other.f))
{
}

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType> &umuqFunction<DataType, FunctionType>::operator=(umuqFunction<DataType, FunctionType> &&other)
{
  name = other.name;
  numParams = other.numParams;
  params = std::move(other.params);
  f = std::move(other.f);

  return *this;
}

template <typename DataType, class FunctionType>
std::string const umuqFunction<DataType, FunctionType>::getName() const
{
  return name;
}

template <typename DataType, class FunctionType>
void umuqFunction<DataType, FunctionType>::setName(char const *Name)
{
  name = std::string(Name);
}

template <typename DataType, class FunctionType>
umuqFunction<DataType, FunctionType>::operator bool() const noexcept
{
  return f != nullptr;
}

} // namespace umuq

#endif //UMUQ_FUNCTION
