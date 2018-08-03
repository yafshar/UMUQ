#ifndef UMUQ_UMUQFUNCTION_H
#define UMUQ_UMUQFUNCTION_H

/*!\class umuqFunction
 * \brief umuqFunction is a general-purpose polymorphic function wrapper of n variables
 *
 * \tparam T  Data type
 * \tparam F  Function type 
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
													 numParams(0)
{
}

template <typename T, class F>
umuqFunction<T, F>::umuqFunction(int const nDim, char const *Name) : name(Name),
																	 numParams(nDim > 0 ? nDim : 0)
{
}

template <typename T, class F>
umuqFunction<T, F>::umuqFunction(T const *Params, int const NumParams, char const *Name) : name(Name),
																						   numParams(NumParams > 0 ? NumParams : 0),
																						   params(Params, Params + NumParams)
{
}

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
	this->name = other.name;
	this->numParams = other.numParams;
	this->params = std::move(other.params);
	this->f = std::move(other.f);

	return *this;
}

/*!
 * \brief Instances of std::function as f(x) 
 * 
 * \tparam T  IN/OUT data type
 */
template <typename T>
using FUN_x = std::function<T(T const)>;

/*!
 * \brief Instances of std::function as f(*x) 
 * 
 * \tparam T  IN/OUT data type
 */
template <typename T>
using FUN_x_p = std::function<T(T *)>;

/*!
 * \brief Instances of std::function as f(&x) 
 * 
 * \tparam T  OUT data type
 * \tparam V  IN data type
 */
template <typename T, class V>
using FUN_x_v = std::function<T(V const &)>;

/*!
 * \brief Instances of std::function as f(x,y) 
 * 
 * \tparam T IN/OUT data type
 * \tparam Y IN data type of the second variable 
 */
template <typename T, typename Y = T>
using FUN_xy = std::function<T(T const, Y const)>;

/*!
 * \brief Instances of std::function as f(*x,*y) 
 * 
 * \tparam T IN/OUT data type
 * \tparam T IN data type of the second variable
 */
template <typename T, typename Y = T>
using FUN_xy_p = std::function<T(T *, Y *)>;

#endif // UMUQ_FUNCTION_H
