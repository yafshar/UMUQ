#ifndef UMUQ_UMUQFUNCTION_H
#define UMUQ_UMUQFUNCTION_H

/*!\class umuqFunction
 * \brief umuqFunction is a general-purpose polymorphic function wrapper of n variables
 *
 * \tparam T  Data type
 * \tparam F  Function type (wrapped as std::function)
 */
template <typename T, class F>
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
    umuqFunction(T const *Params, int const NumParams, char const *Name = "");

    /*!
     * \brief Construct a new umuqFunction object
     *  
     * \param Params  Input parameters of the Function object
     * \param Name    Function name
     */
    umuqFunction(std::vector<T> const &Params, char const *Name = "");

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
    umuqFunction(umuqFunction<T, F> &&other);

    /*!
     * \brief Move assignment operator
     * 
     */
    umuqFunction<T, F> &operator=(umuqFunction<T, F> &&other);

    /*!
     * \brief Get the Name object
     * 
     * \return std::string const 
     */
    std::string const getName() const;

    /*!
     * \brief Checks whether *this stores a callable function target, i.e. is not empty. 
     * 
     * \return true   If it stores a callable function target at f
     * \return false 
     */
    explicit operator bool() const noexcept;

  private:
    // Make it noncopyable
    umuqFunction(umuqFunction<T, F> const &) = delete;

    // Make it not assignable
    umuqFunction<T, F> &operator=(umuqFunction<T, F> const &) = delete;

  public:
    //! Name of the function
    std::string name;

    //! Number of dimensions
    std::size_t numParams;

    //! Function parameters
    std::vector<T> params;

  public:
    /*!
     * \brief A general-purpose polymorphic function wrapper
     * 
     * \returns the function value
     */
    F f;
};

template <typename T, class F>
umuqFunction<T, F>::umuqFunction(char const *Name) : name(Name),
                                                     numParams(0),
                                                     f(nullptr)
{
}

template <typename T, class F>
umuqFunction<T, F>::umuqFunction(int const nDim, char const *Name) : name(Name),
                                                                     numParams(nDim > 0 ? nDim : 0),
                                                                     f(nullptr)
{
}

template <typename T, class F>
umuqFunction<T, F>::umuqFunction(T const *Params, int const NumParams, char const *Name) : name(Name),
                                                                                           numParams(NumParams > 0 ? NumParams : 0),
                                                                                           params(Params, Params + NumParams),
                                                                                           f(nullptr)
{
}

template <typename T, class F>
umuqFunction<T, F>::umuqFunction(std::vector<T> const &Params, char const *Name) : name(Name),
                                                                                   numParams(Params.size()),
                                                                                   params(Params),
                                                                                   f(nullptr)
{
}

template <typename T, class F>
umuqFunction<T, F>::~umuqFunction() {}

template <typename T, class F>
umuqFunction<T, F>::umuqFunction(umuqFunction<T, F> &&other) : name(other.name),
                                                               numParams(other.numParams),
                                                               params(std::move(other.params)),
                                                               f(std::move(other.f))
{
}

template <typename T, class F>
umuqFunction<T, F> &umuqFunction<T, F>::operator=(umuqFunction<T, F> &&other)
{
    name = other.name;
    numParams = other.numParams;
    params = std::move(other.params);
    f = std::move(other.f);

    return *this;
}

template <typename T, class F>
std::string const umuqFunction<T, F>::getName() const
{
  return name;
}

template <typename T, class F>
umuqFunction<T, F>::operator bool() const noexcept
{
  return f != nullptr;
}

#endif //UMUQ_FUNCTION
