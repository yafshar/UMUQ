#ifndef UMUQ_MULTIMIN_FUNCTION_H
#define UMUQ_MULTIMIN_FUNCTION_H

/*! \brief The goal is finding minima of arbitrary multidimensional functions.
 *  \ingroup multimin_Module
 */

/*! \class multimin_function
  * \brief Defines a general function of n variables
  * 
  * \tparam T    Data type
  * \tparam TMF  Multimin function type
  */
template <typename T, class TMF>
class multimin_function
{
  public:
	/*!
	 * \brief Construct a new multimin function object
	 * 
	 */
	multimin_function();

	/*!
	 * \brief Construct a new multimin function object
	 * 
	 * \param nIn The dimension of the system, number of components of the vectors x
	 */
	explicit multimin_function(int const nIn);

	/*!
	 * \brief Function f
	 * 
	 * \param x  Input data
	 * 
	 * \return  f Function value at x 
	 */
	T f(T const *x);

	// The dimension of the system, i.e. the number of components of the vectors x.
	std::size_t n;

  private:
	friend TMF;
};

/*!
 * \brief Construct a new multimin function object
 * 
 */
template <typename T, class TMF>
multimin_function<T, TMF>::multimin_function() {}

/*!
 * \brief Construct a new multimin function object
 * 
 * \param nIn The dimension of the system, number of components of the vectors x
 */
template <typename T, class TMF>
multimin_function<T, TMF>::multimin_function(int const nIn) : n(nIn) {}

/*!
 * \brief Function f
 * 
 * \param x   Input data
 * \return  f Function value at x 
 */
template <typename T, class TMF>
T multimin_function<T, TMF>::f(T const *x)
{
	return static_cast<TMF *>(this)->f(x);
}

/*! \class multimin_fminimizer_type
  * \brief This class specifies minimization algorithms which do not use gradients
  * 
  * \ref multimin_nmsimplex & \ref multimin_nmsimplex2
  * 
  * These methods use the Simplex algorithm of Nelder and Mead.
  * Starting from the initial vector, the algorithm
  * constructs an additional n vectors \f$ p_i \f$ using the step size vector \f$ s = step_{size} \f$.
  * 
  * \tparam T     data type
  * \tparam TMFMT multimin function minimizer type
  * \tparam TMF   multimin function type
  */
template <typename T, class TMFMT, class TMF>
class multimin_fminimizer_type
{
  public:
	bool alloc(std::size_t n)
	{
		return static_cast<TMFMT *>(this)->alloc(n);
	}

	bool set(TMF *tmf, T const *x, T *size, T const *step_size)
	{
		return static_cast<TMFMT *>(this)->set(tmf, x, size, step_size);
	}

	bool iterate(TMF *tmf, T const *x, T *size, T *fval)
	{
		return static_cast<TMFMT *>(this)->iterate(tmf, x, size, fval);
	}

	void free()
	{
		static_cast<TMFMT *>(this)->free();
	}

	const char *name;

  private:
	friend TMFMT;
};

/*! \class multimin_fminimizer
  * \brief This class is for minimizing functions without derivatives.
  * 
  * \tparam T     data type
  * \tparam TMFMT multimin differentiable function minimizer type
  * \tparam TMF   multimin differentiable function type
  */
template <typename T, class TMFMT, class TMF>
class multimin_fminimizer
{
  public:
	multimin_fminimizer() : type(nullptr), f(nullptr), x(nullptr) {}
	~multimin_fminimizer() { free(); }

	/*!
     * \brief alloc
     * 
     * \param Ttype pointer to \a multimin_fminimizer_type object 
     * \param n_ size of array
     * 
     * \returns true if everything goes OK
     */
	bool alloc(TMFMT *Ttype, std::size_t n_)
	{
		n = n_;
		type = Ttype;

		try
		{
			x = new T[n]();
		}
		catch (std::bad_alloc &e)
		{
			UMUQFAILRETURN("Failed to allocate memory");
		}

		if (!type->alloc(n))
		{
			free();
			return false;
		}

		return true;
	}

	/*!
     * \brief set
     * 
     * \param f_        
     * \param x         input array
     * \param step_size step size
     *  
     * returns 
     */
	bool set(TMF *f_, T const *x_, T const *step_size)
	{
		if (n != f_->n)
		{
			UMUQFAILRETURN("Function incompatible with solver size!");
		}

		//set the pointer
		f = f_;

		//copy array x_ to array x
		std::copy(x_, x_ + n, x);

		return type->set(f, x, &size, step_size);
	}

	/*!
     * \brief destructor
     * 
     */
	void free()
	{
		if (type != nullptr)
		{
			type->free();
			type = nullptr;
		}

		if (f != nullptr)
		{
			f = nullptr;
		}

		if (x != nullptr)
		{
			delete[] x;
			x = nullptr;
		}

		n = 0;
	}

	/*!
     * \brief iterate
     * 
     */
	bool iterate()
	{
		return type->iterate(f, x, &size, &fval);
	}

	/*!
     * \brief name
     * \returns the name of the minimization type
     */
	const char *name()
	{
		return type->name;
	}

	/*!
     * \brief get function x
     * \returns x
     */
	T *get_x()
	{
		return x;
	}

	/*!
     * \brief minimum
     * \returns the minimum
     */
	T minimum()
	{
		return fval;
	}

	/*!
     * \brief get_size
     * \returns the size
     */
	T get_size()
	{
		return size;
	}

  public:
	//Multi dimensional part
	TMFMT *type;
	TMF *f;

	T fval;

	T *x;

	T size;

	std::size_t n;
};

/*! \class function_fdf
  * \brief Definition of an arbitrary differentiable function
  *  
  * \tparam T   data type
  * \tparam TFD differentiable function type
  */
template <typename T, class TFD>
class function_fdf
{
  public:
	function_fdf() {}
	function_fdf(std::size_t n_) : n(n_) {}

	T f(T const x)
	{
		return static_cast<TFD *>(this)->f(x);
	}

	T df(T const x)
	{
		return static_cast<TFD *>(this)->df(x);
	}

	void fdf(T const x, T *f_, T *df_)
	{
		static_cast<TFD *>(this)->fdf(x, f_, df_);
	}

	std::size_t n;

  private:
	friend TFD;
};

/*! \class multimin_function_fdf
  * \brief Definition of an arbitrary differentiable function with vector input and parameters
  *  
  * \tparam T      data type
  * \tparam TMFD   multimin differentiable function type
  */
template <typename T, class TMFD>
class multimin_function_fdf
{
  public:
	multimin_function_fdf() {}
	multimin_function_fdf(std::size_t n_) : n(n_) {}

	T f(T const *x)
	{
		return static_cast<TMFD *>(this)->f(x);
	}

	T df(T const *x, T *df_)
	{
		return static_cast<TMFD *>(this)->df(x, df_);
	}

	void fdf(T const *x, T *f_, T *df_)
	{
		static_cast<TMFD *>(this)->fdf(x, f_, df_);
	}

	std::size_t n;

  private:
	friend TMFD;
};

/*! \class multimin_fdfminimizer_type
  * \brief differentiable function minimizer type
  * 
  * \tparam T      data type
  * \tparam TMFDMT multimin differentiable function minimizer type
  * \tparam TMFD   multimin differentiable function type
  */
template <typename T, class TMFDMT, class TMFD>
class multimin_fdfminimizer_type
{
  public:
	bool alloc(std::size_t n)
	{
		return static_cast<TMFDMT *>(this)->alloc(n);
	}

	bool set(TMFD *tmfd, T const *x, T *f, T *gradient, T step_size, T tol)
	{
		return static_cast<TMFDMT *>(this)->set(tmfd, x, f, gradient, step_size, tol);
	}

	bool iterate(TMFD *tmfd, T *x, T *f, T *gradient, T *dx)
	{
		return static_cast<TMFDMT *>(this)->iterate(tmfd, x, f, gradient, dx);
	}

	bool restart()
	{
		return static_cast<TMFDMT *>(this)->restart();
	}

	void free()
	{
		static_cast<TMFDMT *>(this)->free();
	}

	const char *name;

  private:
	friend TMFDMT;
};

/*! \class multimin_fdfminimizer
  * \brief This class is for minimizing functions using derivatives. 
  * 
  * \tparam T      data type
  * \tparam TMFDMT multimin differentiable function minimizer type
  * \tparam TMFD   multimin differentiable function type
  */
template <typename T, class TMFDMT, class TMFD>
class multimin_fdfminimizer
{
  public:
	multimin_fdfminimizer() : type(nullptr), fdf(nullptr), x(nullptr), gradient(nullptr), dx(nullptr) {}
	~multimin_fdfminimizer() { free(); }

	/*!
     * \brief alloc
     * 
     * \param Ttype pointer to \a multimin_fdfminimizer_type object 
     * \param n_ size of array
     * 
     * \returns true if everything goes OK
     */
	bool alloc(TMFDMT *Ttype, std::size_t n_)
	{
		n = n_;
		type = Ttype;

		try
		{
			x = new T[n]();
			//set to zero
			gradient = new T[n]();
			//set to zero
			dx = new T[n]();
		}
		catch (std::bad_alloc &e)
		{
			UMUQFAILRETURN("Failed to allocate memory!");
		}

		if (!type->alloc(n))
		{
			free();
			return false;
		}

		return true;
	}

	/*!
     * \brief set
     * 
     * \param mfdf      pointer to an arbitrary differentiable real-valued function
     * \param x         input array
     * \param n_        size of array n_  
     * \param step_size step size
     * \param tol       tol
     *  
     * returns true if everything goes OK
     */
	bool set(TMFD *fdf_, T const *x_, T step_size, T tol)
	{
		if (n != fdf_->n)
		{
			UMUQFAILRETURN("Function incompatible with solver size!");
		}

		//set the pointer
		fdf = fdf_;

		//copy array x_ to array x
		std::copy(x_, x_ + n, x);

		//set dx to zero
		std::fill(dx, dx + n, T{});

		return type->set(fdf, x, &f, gradient, step_size, tol);
	}

	/*!
     * \brief iterate
     * 
     */
	bool iterate()
	{
		return type->iterate(fdf, x, &f, gradient, dx);
	}

	/*!
     * \brief restart
     * 
     */
	bool restart()
	{
		return type->restart();
	}

	/*!
     * \brief name
     * \returns the name of the minimization type
     */
	const char *name()
	{
		return type->name;
	}

	/*!
     * \brief minimum
     * \returns the minimum
     */
	T minimum()
	{
		return f;
	}

	/*!
     * \brief get function x
     * \returns x
     */
	T *get_x()
	{
		return x;
	}

	/*!
     * \brief get function dx
     * \returns dx
     */
	T *get_dx()
	{
		return dx;
	}

	/*!
     * \brief get function x
     * \returns x
     */
	T *get_gradient()
	{
		return gradient;
	}

	/*!
     * \brief destructor
     * 
     */
	void free()
	{
		if (type != nullptr)
		{
			type->free();
			type = nullptr;
		}

		if (fdf != nullptr)
		{
			fdf = nullptr;
		}

		if (x != nullptr)
		{
			delete[] x;
			x = nullptr;
		}

		if (gradient != nullptr)
		{
			delete[] gradient;
			gradient = nullptr;
		}

		if (dx != nullptr)
		{
			delete[] dx;
			dx = nullptr;
		}

		n = 0;
	}

  private:
	// multi dimensional part
	TMFDMT *type;
	TMFD *fdf;

	T f;

	T *x;
	T *gradient;
	T *dx;

	std::size_t n;
};

template <typename T>
int multimin_test_gradient(T const *g, std::size_t const n, T const epsabs)
{
	if (epsabs < T{})
	{
		UMUQWARNING("Absolute tolerance is negative!");
		//fail
		return -1;
	}

	//Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
	T norm(0);
	std::for_each(g, g + n, [&](T const g_i) { norm += g_i * g_i; });

	if (std::sqrt(norm) < epsabs)
	{
		//success
		return 0;
	}

	//continue
	return 1;
}

template <typename T>
int multimin_test_size(T const size, T const epsabs)
{
	if (epsabs < 0)
	{
		UMUQWARNING("Absolute tolerance is negative!");
		//fail
		return -1;
	}

	if (size < epsabs)
	{
		//success
		return 0;
	}

	//continue
	return 1;
}

template <typename T, class TMF>
bool multimin_diff(TMF *f, T const *x, T *g)
{
	std::size_t n = f->n;

	T const h = std::sqrt(std::numeric_limits<T>::epsilon());

	T *x1;

	try
	{
		x1 = new T[n];
	}
	catch (std::bad_alloc &e)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}

	std::copy(x, x + n, x1);

	for (std::size_t i = 0; i < n; i++)
	{
		T fl;
		T fh;

		T xi = x[i];

		T dx = std::abs(xi) * h;
		if (dx <= T{})
		{
			dx = h;
		}

		x1[i] = xi + dx;

		fh = f->f(x1);

		x1[i] = xi - dx;

		fl = f->f(x1);

		x1[i] = xi;

		g[i] = (fh - fl) / (2 * dx);
	}

	delete[] x1;

	return true;
}

#endif
