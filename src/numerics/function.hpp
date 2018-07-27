#ifndef UMUQ_FUNCTION_H
#define UMUQ_FUNCTION_H

/*!\class umuqfunction
 * \brief umuqfunction is a general-purpose polymorphic function wrapper of n variables
 *
 * \tparam T  Data type
 * \tparam F  Function type 
 */
template <typename T, class F>
class umuqfunction
{
  public:
	/*!
     * \brief Construct a new umuqfunction object
     * 
     * \param Name  Function name
     */
	umuqfunction(char const *Name = "");

	/*!
     * \brief Construct a new umuqfunction object
     * 
     * \param nDim  Number of dimensions (Number of parameters) 
     * \param Name  Function name
     */
	umuqfunction(int const nDim, char const *Name = "");

	/*!
     * \brief Construct a new umuqfunction object
     * 
     * \param Params    Input parameters of the Function object
     * \param NumParams Number of dimensions (Number of parameters) 
     * \param Name      Function name
     */
	umuqfunction(T const *Params, int const NumParams, char const *Name = "");

	/*!
	 * \brief Construct a new umuqfunction object
	 *  
	 * \param Params  Input parameters of the Function object
	 * \param Name    Function name
	 */
	umuqfunction(std::vector<T> const &Params, char const *Name = "");

	/*!
	 * \brief Move constructor, Construct a new umuqfunction object
	 * 
	 * \param other umuqfunction object
	 */
	umuqfunction(umuqfunction<T, F> &&other);

	/*!
	 * \brief Move assignment operator
	 * 
	 */
	umuqfunction<T, F> &operator=(umuqfunction<T, F> &&other);

  private:
	// Make it noncopyable
	umuqfunction(umuqfunction<T, F> const &) = delete;

	// Make it not assignable
	umuqfunction<T, F> &operator=(umuqfunction<T, F> const &) = delete;

  public:
	//! Name of the function
	std::string name;

	//! Number of dimensions
	std::size_t numParams;

	//! Function parameters
	std::vector<T> params;

  public:
	//! A general-purpose polymorphic function wrapper
	F f;
};

template <typename T, class F>
umuqfunction<T, F>::umuqfunction(char const *Name) : name(Name),
													 numParams(0)
{
}

template <typename T, class F>
umuqfunction<T, F>::umuqfunction(int const nDim, char const *Name) : name(Name),
																	 numParams(nDim > 0 ? nDim : 0)
{
}

template <typename T, class F>
umuqfunction<T, F>::umuqfunction(T const *Params, int const NumParams, char const *Name) : name(Name),
																						   numParams(NumParams > 0 ? NumParams : 0),
																						   params(Params, Params + NumParams)
{
}

template <typename T, class F>
umuqfunction<T, F>::umuqfunction(umuqfunction<T, F> &&other) : name(other.name),
															   numParams(other.numParams),
															   params(std::move(other.params)),
															   f(std::move(other.f))
{
}

template <typename T, class F>
umuqfunction<T, F> &umuqfunction<T, F>::operator=(umuqfunction<T, F> &&other)
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
