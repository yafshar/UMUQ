#ifndef UMHBM_MULTIMIN_LINEAR_WRAPPER_H
#define UMHBM_MULTIMIN_LINEAR_WRAPPER_H

#include "multimin.hpp"

/*! \class function_fdf
  * \brief Definition of an arbitrary differentiable function with vector input and parameters
  *  
  * \tparam T      data type
  * \tparan TMFD   multimin differentiable function type
  */
template <typename T, class TMFD>
class wrapper_t
{
  public:
	/*!
     * \brief 
     * 
     */
	void prepare(multimin_function_fdf<T, TMFD> *fdf_, T const *x_, T const f_, T const *g_, T const *p_, T *x_alpha_, T *g_alpha_)
	{
		fdf = fdf_;
		n = fdf->n;
		fdf_linear.n = n;
		
		x = x_;
		f_alpha = f_;
		g = g_;
		p = p_;
		x_alpha = x_alpha_;
		g_alpha = g_alpha_;
		x_cache_key = (T)0;
		f_cache_key = (T)0;
		g_cache_key = (T)0;
		df_cache_key = (T)0;

		std::copy(x, x + n, x_alpha);
		std::copy(g, g + n, g_alpha);

		df_alpha = slope();
	}

	/*!
     * \brief 
     * 
     */
	void update_position(T alpha, T *x_, T *f, T *g_)
	{
		//Ensure that everything is fully cached
		{
			T f_alpha_;
			T df_alpha_;
			fdf_linear.fdf(alpha, &f_alpha_, &df_alpha_);
		};

		*f = f_alpha;

		std::copy(x_alpha, x_alpha + n, x_);
		std::copy(g_alpha, g_alpha + n, g_);
	}

	/*!
     * \brief 
     * 
     */
	void change_direction()
	{
		//Convert the cache values from the end of the current minimization
		//to those needed for the start of the next minimization, alpha=0

		//The new x_alpha for alpha=0 is the current position
		std::copy(x, x + n, x_alpha);

		x_cache_key = (T)0;

		//The function value does not change
		f_cache_key = (T)0;

		//The new g_alpha for alpha=0 is the current gradient at the endpoint
		std::copy(g, g + n, g_alpha);

		g_cache_key = (T)0;

		//Calculate the slope along the new direction vector, p
		df_alpha = slope();
		df_cache_key = (T)0;
	}

	/*!
     * \brief 
     * 
     */
	void moveto(T const alpha)
	{
		//using previously cached position
		if (alpha == x_cache_key)
		{
			return;
		}

		// set x_alpha = x + alpha * p
		std::copy(x, x + n, x_alpha);

		for (size_t i = 0; i < n; i++)
		{
			x_alpha[i] += alpha * p[i];
		}

		x_cache_key = alpha;
	}

	/*!
	 * \brief compute gradient . direction
	 */
	inline T slope()
	{
		T df(0);
		for (size_t i = 0; i < n; i++)
		{
			df += g_alpha[i] * p[i];
		}

		return df;
	}

	class wrap : public function_fdf<T, wrap>
	{
	  public:
		/*!
         * \brief 
         * 
         */
		T f(T const alpha)
		{
			//using previously cached f(alpha)
			if (alpha == f_cache_key)
			{
				return f_alpha;
			}

			moveto(alpha);

			f_alpha = fdf->f(x_alpha);

			f_cache_key = alpha;

			return f_alpha;
		}

		/*!
         * \brief 
         * 
         */
		T df(T const alpha)
		{
			//using previously cached df(alpha)
			if (alpha == df_cache_key)
			{
				return df_alpha;
			}

			moveto(alpha);

			if (alpha != g_cache_key)
			{
				fdf->df(x_alpha, g_alpha);

				g_cache_key = alpha;
			}

			df_alpha = slope();

			df_cache_key = alpha;

			return df_alpha;
		}

		/*!
         * \brief 
         * 
         */
		void fdf(T const alpha, T *f, T *df)
		{

			//Check for previously cached values
			if (alpha == f_cache_key && alpha == df_cache_key)
			{
				*f = f_alpha;
				*df = df_alpha;
				return;
			}

			if (alpha == f_cache_key || alpha == df_cache_key)
			{
				*f = fdf_linear.f(alpha);
				*df = fdf_linear.df(alpha);
				return;
			}

			moveto(alpha);

			fdf->fdf(x_alpha, &f_alpha, g_alpha);

			f_cache_key = alpha;
			g_cache_key = alpha;

			df_alpha = slope();
			df_cache_key = alpha;

			*f = f_alpha;
			*df = df_alpha;
		}
	};

  public:
	wrap fdf_linear;

  private:
	multimin_function_fdf<T, TMFD> *fdf;

	// fixed values
	T const *x;
	T const *g;
	T const *p;

	// cached values, for x(alpha) = x + alpha * p
	T f_alpha;
	T df_alpha;
	T *x_alpha;
	T *g_alpha;

	// cache "keys"
	T f_cache_key;
	T df_cache_key;
	T x_cache_key;
	T g_cache_key;

	size_t n;
};

#endif
