#ifndef UMHBM_MULTIMIN_VECTOR_BFGS2_H
#define UMHBM_MULTIMIN_VECTOR_BFGS2_H

#include "multimin.hpp"

#include "multimin_linear_minimize.hpp"
#include "multimin_linear_wrapper.hpp"

/*! \class vector_bfgs2
  * \brief Limited memory Broyden-Fletcher-Goldfarb-Shanno method
  * Fletcher's implementation of the BFGS method,
  * using the line minimization algorithm from from R.Fletcher,
  * "Practical Methods of Optimization", Second Edition, ISBN
  * 0471915475.  Algorithms 2.6.2 and 2.6.4.
  * 
  * 
  * \tparam T      data type
  * \tparan TMFD   multimin differentiable function type
  */
template <typename T, class TMFD>
class vector_bfgs2 : public multimin_fdfminimizer_type<T, vector_bfgs2<T, TMFD>, TMFD>
{
  public:
	/*!
     * \brief constructor
     * 
     * \param name name of the differentiable function minimizer type (default "vector_bfgs2")
     */
	vector_bfgs2(const char *name_ = "vector_bfgs2") : name(name_),
													   p(nullptr),
													   x0(nullptr),
													   g0(nullptr),
													   dx0(nullptr),
													   dg0(nullptr),
													   x_alpha(nullptr),
													   g_alpha(nullptr) {}

	/*!
     * \brief destructor
     */
	~vector_bfgs2() { free(); }

	/*!
     * \brief allocate space for data type T
     * 
     * \param n_ size of array
     * 
     * \returns false if there is insufficient memory to create data array 
     */
	bool alloc(size_t n_)
	{
		n = n_;
		try
		{
			p = new T[n]();
			x0 = new T[n]();
			g0 = new T[n]();
			dx0 = new T[n]();
			dg0 = new T[n]();
			x_alpha = new T[n]();
			g_alpha = new T[n]();
		}
		catch (std::bad_alloc &e)
		{
			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
			return false;
		}

		return true;
	}

	/*!
     * \brief set
     * 
     * \param fdf differentiable function minimizer
     * \param x   array of data
     * \param f   
     * \param gradient
     * \param step_size
     * \param tol
     */
	bool set(TMFD *fdf, T const *x, T *f, T *gradient, T step_size, T tol)
	{
		iter = 0;
		step = step_size;
		delta_f = 0;

		fdf->fdf(x, f, gradient);

		// Use the gradient as the initial direction
		std::copy(x, x + n, x0);
		std::copy(gradient, gradient + n, g0);

		{
			//Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
			T s(0);
			std::for_each(g0, g0 + n, [&](T const g) { s += g * g; });
			g0norm = std::sqrt(s);
		}

		std::copy(gradient, gradient + n, p);

		{
			T alpha = -1 / g0norm;
			std::for_each(p, p + n, [&](T &p_) { p_ *= alpha; });
		}

		{
			//Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
			T s(0);
			std::for_each(p, p + n, [&](T const p_) { s += p_ * p_; });
			//should be 1
			pnorm = std::sqrt(s);
		}

		fp0 = -g0norm;

		//Prepare the wrapper
		wrap.prepare(fdf, x0, *f, g0, p, x_alpha, g_alpha);

		//Prepare 1d minimization parameters
		rho = 0.01;
		sigma = tol;
		tau1 = 9;
		tau2 = 0.05;
		tau3 = 0.5;

		//Use cubic interpolation where possible
		order = 3;

		return true;
	}

	void free()
	{
		delete[] p;
		p = nullptr;
		delete[] x0;
		x0 = nullptr;
		delete[] g0;
		g0 = nullptr;
		delete[] dx0;
		dx0 = nullptr;
		delete[] dg0;
		dg0 = nullptr;
		delete[] g_alpha;
		g_alpha = nullptr;
		delete[] x_alpha;
		x_alpha = nullptr;
	}

	bool restart()
	{
		iter = 0;
		return true;
	}

	bool iterate(TMFD *fdf, T *x, T *f, T *gradient, T *dx)
	{

		if (pnorm == (T)0 || g0norm == (T)0 || fp0 == (T)0)
		{
			//set dx to zero
			std::fill(dx, dx + n, (T)0);

			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << " The minimizer is unable to improve on its current estimate, either due" << std::endl;
			std::cerr << " to numerical difficulty or because a genuine local minimum has been reached." << std::endl;
			return false;
		}

		T alpha(0);
		T alpha1;

		T pg;
		T dir;

		T f0 = *f;

		if (delta_f < 0)
		{
			T del = std::max(-delta_f, 10 * std::numeric_limits<T>::epsilon() * std::abs(f0));
			alpha1 = std::min((T)1, 2 * del / fp0);
		}
		else
		{
			alpha1 = std::abs(step);
		}

		//Line minimization, with cubic interpolation (order = 3)

		bool status = minimize<T, function_fdf<T, wrapper_t<T, TMFD>::wrap>>(&wrap.fdf_linear, rho, sigma,
																			 tau1, tau2, tau3, order,
																			 alpha1, &alpha);
		if (status != true)
		{
			return false;
		}

		wrap.update_position(alpha, x, f, gradient);

		delta_f = *f - f0;

		//Choose a new direction for the next step
		{
			//This is the BFGS update:
			// /f$ p' = g1 - A dx - B dg /f$
			// /f$ A = - (1+ dg.dg/dx.dg) B + dg.g/dx.dg /f$
			// /f$ B = dx.g/dx.dg /f$

			// /f$ dx0 = x - x0 /f$
			std::copy(x, x + n, dx0);

			for (size_t i = 0; i < n; i++)
			{
				dx0[i] -= x0[i];
			}

			//keep a copy
			std::copy(dx0, dx0 + n, dx);

			// /f$ dg0 = g - g0 /f$
			std::copy(gradient, gradient + n, dg0);

			for (size_t i = 0; i < n; i++)
			{
				dg0[i] -= g0[i];
			}

			T dxg(0);
			for (size_t i = 0; i < n; i++)
			{
				dxg += dx0[i] * gradient[i];
			}

			T dgg(0);
			for (size_t i = 0; i < n; i++)
			{
				dgg += dg0[i] * gradient[i];
			}

			T dxdg(0);
			for (size_t i = 0; i < n; i++)
			{
				dxdg += dx0[i] * dg0[i];
			}

			T dgnorm;
			{
				T s(0);
				std::for_each(dg0, dg0 + n, [&](T const d) { s += d * d; });
				dgnorm = std::sqrt(s);
			}

			T A;
			T B;

			if (dxdg != 0)
			{
				B = dxg / dxdg;
				A = -(1.0 + dgnorm * dgnorm / dxdg) * B + dgg / dxdg;
			}
			else
			{
				B = 0;
				A = 0;
			}

			std::copy(gradient, gradient + n, p);

			for (size_t i = 0; i < n; i++)
			{
				p[i] -= A * dx0[i];
			}

			for (size_t i = 0; i < n; i++)
			{
				p[i] -= B * dg0[i];
			}
		}

		std::copy(gradient, gradient + n, g0);
		std::copy(x, x + n, x0);

		{
			T s(0);
			std::for_each(g0, g0 + n, [&](T const g0_) { s += g0_ * g0_; });
			g0norm = std::sqrt(s);
		}

		{
			T s(0);
			std::for_each(p, p + n, [&](T const p_) { s += p_ * p_; });
			pnorm = std::sqrt(s);
		}

		//Update direction and fp0
		pg = (T)0;
		for (size_t i = 0; i < n; i++)
		{
			pg += p[i] * gradient[i];
		}

		dir = (pg >= (T)0) ? -(T)1 : (T)1;

		{
			T alpha = dir / pnorm;
			std::for_each(p, p + n, [&](T &p_) { p_ = alpha * p_; });
		}

		{
			T s(0);
			std::for_each(p, p + n, [&](T const p_) { s += p_ * p_; });
			pnorm = std::sqrt(s);
		}

		fp0 = (T)0;
		for (size_t i = 0; i < n; i++)
		{
			fp0 += g0[i] * p[i];
		}

		wrap.change_direction();

		return true;
	}

  private:
	int iter;

	T step;

	T pnorm;
	T g0norm;
	T delta_f;
	//f'(0) for f(x-alpha*p)
	T fp0;

	T *p;
	T *x0;
	T *g0;
	//work space
	T *dx0;		//
	T *dg0;		//
	T *x_alpha; //
	T *g_alpha; //

	//wrapper function
	wrapper_t<T, TMFD> wrap;

	//minimization parameters
	T rho;
	T sigma;
	T tau1;
	T tau2;
	T tau3;

	int order;

	size_t n;
}

#endif