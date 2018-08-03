#ifndef UMUQ_MULTIMIN_NSIMPLEX2RAND_H
#define UMUQ_MULTIMIN_NSIMPLEX2RAND_H

/*! \class nmsimplex2rand
 *  \ingroup multimin_Module
 * 
 * \brief The Simplex method
 * 
 * The Simplex method of Nelder and Mead, also known as the polytope
 * search alogorithm.  Ref: Nelder, J.A., Mead, R., Computer Journal 7
 * (1965) pp. 308-313.
 * 
 * This implementation uses n+1 corner points in the simplex.
 * 
 * \tparam T   data type
 * \tparam TMF multimin function type
 */
template <typename T, class TMF>
class nmsimplex2rand : public multimin_fminimizer_type<T, nmsimplex2rand<T, TMF>, TMF>
{
  public:
	/*!
     * \brief constructor
     * 
     * \param name_ Name of the differentiable function minimizer type (default "nmsimplex2rand")
     */
	nmsimplex2rand(const char *name_ = "nmsimplex2rand") : x1(nullptr),
														   y1(nullptr),
														   ws1(nullptr),
														   ws2(nullptr),
														   center(nullptr),
														   delta(nullptr),
														   xmc(nullptr) { this->name = name_; }

	/*!
     * \brief destructor
     */
	~nmsimplex2rand() { free(); }

	/*!
     * \brief Allocate space for data type T
     * 
     * \param n_ Array size
     * 
     * \returns  False if there is insufficient memory to create data array 
     */
	bool alloc(std::size_t const n_)
	{
		if (n_ <= 0)
		{
			UMUQFAILRETURN("Invalid number of parameters specified!");
		}

		n = n_;
		try
		{
			std::ptrdiff_t const N = (n + 1) * n;
			x1 = new T[N];
			y1 = new T[n + 1];

			ws1 = new T[n];
			ws2 = new T[n];
			center = new T[n];
			delta = new T[n];
			xmc = new T[n];
		}
		catch (std::bad_alloc &e)
		{
			UMUQFAILRETURN("Failed to allocate memory!");
		}

		count = 0;

		return true;
	}

	void free()
	{
		if (x1 != nullptr)
		{
			delete[] x1;
			x1 = nullptr;
		}
		if (y1 != nullptr)
		{
			delete[] y1;
			y1 = nullptr;
		}
		if (ws1 != nullptr)
		{
			delete[] ws1;
			ws1 = nullptr;
		}
		if (ws2 != nullptr)
		{
			delete[] ws2;
			ws2 = nullptr;
		}
		if (center != nullptr)
		{
			delete[] center;
			center = nullptr;
		}
		if (delta != nullptr)
		{
			delete[] delta;
			delta = nullptr;
		}
		if (xmc != nullptr)
		{
			delete[] xmc;
			xmc = nullptr;
		}
	}

	/*!
     * \brief set
     * 
     */
	inline T ran_unif(unsigned long *seed)
	{
		unsigned long s = *seed;
		*seed = (s * 69069 + 1) & 0xffffffffUL;
		return (*seed) / static_cast<T>(4294967296);
	}

	/*!
     * \brief set
     * 
     */
	bool set(TMF *f, T const *x, T *size, T const *step_size)
	{
		//First point is the original x0
		T val = f->f(x);

		if (!std::isfinite(val))
		{
			UMUQFAILRETURN("Non-finite function value encountered!");
		}

		//Copy the elements of the vector x into the 0-th row of the matrix x1
		std::copy(x, x + n, x1);

		y1[0] = val;

		{
			submatrix m(n + 1, n, 1, 0, n, n);

			//Set the elements of the submatrix m to the corresponding elements of the identity matrix
			for (std::size_t i = 0; i < n; i++)
			{
				for (std::size_t j = 0; j < n; j++)
				{
					std::ptrdiff_t const Id = m.ID(i, j);
					x1[Id] = (i == j) ? static_cast<T>(1) : T{};
				}
			}

			//Generate a random orthornomal basis  */
			unsigned long seed = count ^ 0x12345678;

			//warm it up
			ran_unif(&seed);

			//Start with random reflections
			for (std::size_t i = 0; i < n; i++)
			{
				T s = ran_unif(&seed);
				if (s > 0.5)
				{
					std::ptrdiff_t const Id = m.ID(i, i);
					x1[Id] = -static_cast<T>(1);
				}
			}

			//Apply random rotations
			for (std::size_t i = 0; i < n; i++)
			{
				for (std::size_t j = i + 1; j < n; j++)
				{
					//Rotate columns i and j by a random angle
					T const angle = M_2PI * ran_unif(&seed);
					T const c = std::cos(angle);
					T const s = std::sin(angle);

					//Apply a Givens rotation
					for (std::size_t r = 0; r < n; r++)
					{
						std::ptrdiff_t const Id_ci = m.ID(r, i);
						std::ptrdiff_t const Id_c_j = m.ID(r, j);
						T const x_r = x1[Id_ci];
						T const y_r = x1[Id_c_j];
						x1[Id_ci] = c * x_r + s * y_r;
						x1[Id_c_j] = -s * x_r + c * y_r;
					}
				}
			}

			//Scale the orthonormal basis by the user-supplied step_size in
			//each dimension, and use as an offset from the central point x
			for (std::size_t i = 0; i < n; i++)
			{
				T const x_i = x[i];
				T const s_i = step_size[i];

				for (std::size_t j = 0; j < n; j++)
				{
					std::ptrdiff_t const Id_ij = m.ID(i, j);

					x1[Id_ij] *= s_i;
					x1[Id_ij] += x_i;
				}
			}

			//Compute the function values at each offset point
			for (std::size_t i = 0; i < n; i++)
			{
				std::ptrdiff_t const Id = m.ID(i, 0);

				T *r_i = x1 + Id;

				val = f->f(r_i);

				if (!std::isfinite(val))
				{
					UMUQFAILRETURN("Non-finite function value encountered!");
				}

				y1[i + 1] = val;
			}
		}

		compute_center();

		//Initialize simplex size
		*size = compute_size();

		count++;

		return true;
	}

	bool iterate(TMF *f, T *x, T *size, T *fval)
	{
		//Simplex iteration tries to minimize function f value
		//xc and xc2 vectors store tried corner point coordinates
		T *xc = ws1;
		T *xc2 = ws2;

		std::size_t hi(0);
		std::size_t lo(0);
		std::size_t s_hi(1);

		//Get index of highest, second highest and lowest point
		T dhi = y1[0];
		T dlo = y1[0];
		T ds_hi = y1[1];

		T val;
		T val2;

		for (std::size_t i = 1; i < n; i++)
		{
			val = y1[i];
			if (val < dlo)
			{
				dlo = val;
				lo = i;
			}
			else if (val > dhi)
			{
				ds_hi = dhi;
				s_hi = hi;
				dhi = val;
				hi = i;
			}
			else if (val > ds_hi)
			{
				ds_hi = val;
				s_hi = i;
			}
		}

		//Ty reflecting the highest value point
		val = try_corner_move(-static_cast<T>(1), hi, xc, f);

		if (std::isfinite(val) && val < y1[lo])
		{
			//Reflected point becomes lowest point, try expansion
			val2 = try_corner_move(-static_cast<T>(2), hi, xc2, f);

			if (std::isfinite(val2) && val2 < y1[lo])
			{
				update_point(hi, xc2, val2);
			}
			else
			{
				update_point(hi, xc, val);
			}
		}
		else if (!std::isfinite(val) || val > y1[s_hi])
		{
			//Reflection does not improve things enough, or we got a non-finite function value

			if (std::isfinite(val) && val <= y1[hi])
			{

				//If trial point is better than highest point, replace highest point
				update_point(hi, xc, val);
			}

			//Try one dimensional contraction
			val2 = try_corner_move(static_cast<T>(0.5), hi, xc2, f);

			if (std::isfinite(val2) && val2 <= y1[hi])
			{
				update_point(hi, xc2, val2);
			}
			else
			{
				//Contract the whole simplex in respect to the best point
				if (!contract_by_best(lo, xc, f))
				{
					UMUQFAILRETURN("contract_by_best failed!");
				}
			}
		}
		else
		{
			//Trial point is better than second highest point. Replace highest point by it
			update_point(hi, xc, val);
		}

		//Return lowest point of simplex as x
		lo = static_cast<std::size_t>(std::distance(y1, std::min_element(y1, y1 + n)));

		//Copy the elements of the lo-th row of the matrix x1 into the vector x
		{
			std::ptrdiff_t const Id = lo * n;
			std::copy(x1 + Id, x1 + Id + n, x);
		}

		*fval = y1[lo];

		//Update simplex size
		//Recompute if accumulated error has made size invalid
		*size = (S2 > 0) ? std::sqrt(S2) : compute_size();

		return true;
	}

	/*!
	 * \brief Moves a simplex corner scaled by coeff
	 * 
	 * Moves a simplex corner scaled by coeff (negative value represents
	 * mirroring by the middle point of the "other" corner points)
	 * and gives new corner in xc and function value at xc as a
	 * return value
	 * 
	 * \param coeff    coefficient to scale a simplex corner
	 * \param corner   simplex corner
	 * \param xc       input point
	 * \param f        function
	 * \return         function value at xc
	 */
	T try_corner_move(T const coeff, std::size_t corner, T *xc, TMF *f)
	{
		/* 
		 * \f$ N = n + 1 \f$
		 * \f$ xc = (1-coeff)*((N)/(N-1)) * center(all) + ((N*coeff-1)/(N-1))*x_corner \f$
		 */
		T const alpha = (1 - coeff) * (n + 1) / static_cast<T>(n);
		T const beta = ((n + 1) * coeff - 1) / static_cast<T>(n);

		std::copy(center, center + n, xc);
		std::for_each(xc, xc + n, [&](T &x_i) { x_i *= alpha; });

		std::ptrdiff_t const Id = corner * n;
		T *row = x1 + Id;

		for (std::size_t i = 0; i < n; i++)
		{
			xc[i] += beta * row[i];
		}

		return f->f(xc);
	}

	void update_point(std::size_t const i, T const *x, T const val)
	{
		std::ptrdiff_t const Id = i * n;
		T *x_orig = x1 + Id;

		//Compute \f$ delta = x - x_orig \f$
		std::copy(x, x + n, delta);

		for (std::size_t j = 0; j < n; j++)
		{
			delta[j] -= x_orig[j];
		}

		//Compute \f$ xmc = x_orig - c \f$
		std::copy(x_orig, x_orig + n, xmc);

		for (std::size_t j = 0; j < n; j++)
		{
			xmc[j] -= center[j];
		}

		T const N = static_cast<T>(n + 1);

		//Update size: \f$ S2' = S2 + (2/N) * (x_orig - c).delta + (N-1)*(delta/N)^2 \f$
		{
			T dsq(0);
			std::for_each(delta, delta + n, [&](T const d_i) { dsq += d_i * d_i; });

			T sum(0);
			for (std::size_t j = 0; j < n; j++)
			{
				sum += xmc[j] * delta[j];
			}

			S2 += (static_cast<T>(2) / N) * sum + (static_cast<T>(n) / N) * (dsq / N);
		}

		//Update center:  \f$ c' = c + (x - x_orig) / N \f$
		{
			T const alpha = static_cast<T>(1) / N;

			for (std::size_t j = 0; j < n; j++)
			{
				center[j] -= alpha * x_orig[j];
			}

			for (std::size_t j = 0; j < n; j++)
			{
				center[j] += alpha * x[j];
			}
		}

		//Copy the elements of the vector x into the i-th row of the matrix x1
		std::copy(x, x + n, x_orig);

		y1[i] = val;
	}

	/*!
     * \brief Contracts the simplex in respect to best valued corner
     * 
     * Function contracts the simplex in respect to best valued corner. 
     * That is, all corners besides the best corner are moved.
     * (This function is rarely called in practice, since it is the last
     * choice, hence not optimized) 
     * 
     * \param best    best valued corner
     * \param xc      work space 
     * \param f       function
     * \return true 
     * \return false  Fining at least one bad function value
     */
	bool contract_by_best(std::size_t best, T *xc, TMF *f)
	{
		//The xc vector is simply work space here
		T newval;

		for (std::size_t i = 0; i < n + 1; i++)
		{
			if (i != best)
			{
				std::ptrdiff_t Id = i * n;
				std::ptrdiff_t Idb = best * n;

				for (std::size_t j = 0; j < n; j++, Id++, Idb++)
				{
					newval = static_cast<T>(0.5) * (x1[Id] + x1[Idb]);
					x1[Id] = newval;
				}

				//Evaluate function in the new point
				Id = i * n;

				//Copy the elements of the i-th row of the matrix x1 into the vector xc
				std::copy(x1 + Id, x1 + Id + n, xc);

				newval = f->f(xc);

				y1[i] = newval;

				/*
				 * Notify caller that we found at least one bad function value.
				 * we finish the contraction (and do not abort) to allow the user
				 * to handle the situation
				 */
				if (!std::isfinite(newval))
				{
					UMUQFAILRETURN("The iteration encountered a singular point where the function or its derivative evaluated to Inf or NaN!");
				}
			}
		}

		//We need to update the centre and size as well
		compute_center();
		compute_size();

		return true;
	}

	/*!
     * \brief Calculates the center of the simplex and stores in center
     * 
     * \return true 
     */
	bool compute_center()
	{
		//Calculates the center of the simplex and stores in center
		std::fill(center, center + n, T{});

		for (std::size_t i = 0; i < n + 1; i++)
		{
			std::ptrdiff_t const Id = i * n;
			T *row = x1 + Id;

			for (std::size_t j = 0; j < n; j++)
			{
				center[j] += row[j];
			}
		}

		{
			T const alpha = static_cast<T>(1) / static_cast<T>(n + 1);
			std::for_each(center, center + n, [&](T &c_i) { c_i *= alpha; });
		}

		return true;
	}

	/*!
     * \brief Calculates simplex size 
     * 
     * \return Store squared size
     */
	T compute_size()
	{
		//Calculates simplex size as rms sum of length of vectors
		//from simplex center to corner points:
		//\f$ sqrt( sum ( || y - y_middlepoint ||^2 ) / n ) \f$
		T ss(0);
		for (std::size_t i = 0; i < n + 1; i++)
		{
			//Copy the elements of the i-th row of the matrix x1 into the vector s
			std::ptrdiff_t const Id = i * n;
			std::copy(x1 + Id, x1 + Id + n, ws1);

			for (std::size_t j = 0; j < n; j++)
			{
				ws1[j] -= center[j];
			}

			T t(0);
			std::for_each(ws1, ws1 + n, [&](T const w_i) { t += w_i * w_i; });

			//squared size
			ss += t;
		}

		//Store squared size
		S2 = ss / static_cast<T>(n + 1);

		return std::sqrt(S2);
	}

  private:
	//! Simplex corner points (Matrix of size \f$ (n+1) \times n \f$
	T *x1;

	//! Function value at corner points with size \f$ (n+1) \f$
	T *y1;

	//! Workspace 1 for algorithm
	T *ws1;

	//! Workspace 2 for algorithm
	T *ws2;

	//! Center of all points
	T *center;

	//! Current step
	T *delta;

	//! x - center (workspace)
	T *xmc;

	//! Store squared size
	T S2;

	//! counter
	unsigned long count;

	//! Array size
	std::size_t n;

  private:
	/*!
     * \return a (pointer to a) row of the data.
     */
	inline T *operator[](std::size_t index) const
	{
		return x1 + index * n;
	}

	/*!
     * Returns memory id of an element in a matrix view of a submatrix of the matrix x1.
     * The upper-left element of the submatrix is the element (k1,k2) of the original 
     * matrix. The submatrix has n1 rows and n2 columns.
     * The physical number of columns in memory given by NC is unchanged.
     * Mathematically, the (i,j)-th element of the new matrix is given by,
     * \f$ ID(i, j)_{(NC, k1, k2, n1, n2)} = [(k1 * NC + k2) + i*NC + j ]   \f$
     * 
     */
	class submatrix
	{
	  public:
		/*!
         * \brief constructor
         * 
         * \param NR  number of rows in the original matrix
         * \param NC_ number of columns in the original matrix
         * \param k1_ row number of the upper-left element of the submatrix
         * \param k2_ column number of the upper-left element of the submatrix
         * \param n1_ submatrix number of rows
         * \param n2_ submatrix number of columns
         */
		submatrix(std::size_t NR, std::size_t NC_, std::size_t k1_, std::size_t k2_, std::size_t n1_, std::size_t n2_)
		{
			if (k1_ > NR || k2_ > NC_ || n1_ > NR || n2_ > NC_)
			{
				std::cerr << "submatrix of size " << n1_ << " " << n2_ << std::endl;
				std::cerr << "from matrix of size " << NR << " " << NC_ << std::endl;
				std::cerr << "start index of " << k1_ << " " << k2_ << std::endl;
				std::cerr << "  " << std::endl;
				UMUQFAIL("Input data overrun the ends of the original matrix!");
			}
			NC = NC_;
			k1 = k1_;
			k2 = k2_;
			n1 = n1_;
			n2 = n2_;
		}

		/*!
         * \brief memory ID of an element in a matrix view of a submatrix of the matrix x1
         * 
         * \Returns memory id of an element in a matrix view of a submatrix of the matrix x1
         */
		inline std::ptrdiff_t ID(std::size_t i, std::size_t j) const
		{
			return k1 * NC + k2 + i * NC + j;
		}

	  private:
		std::size_t NC;
		std::size_t k1;
		std::size_t k2;
		std::size_t n1;
		std::size_t n2;
	};
};

#endif
