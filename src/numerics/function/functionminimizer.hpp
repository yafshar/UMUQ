#ifndef UMUQ_FUNCTIONMINIMIZER_H
#define UMUQ_FUNCTIONMINIMIZER_H

#include "umuqdifferentiablefunction.hpp"

template <typename T, class F>
class functionMinimizer
{
  public:
	explicit functionMinimizer(char const *Name = "");

  private:
  public:
	//! Name of the functionMinimizer
	std::string name;

	//! Function to be used in this minimizer
	umuqFunction<T, F> *fun;

    //! Initial point
	std::vector<T> x; 

	//! Function value
	T fval;
};

template <typename T, class F>
class differentiableFunctionMinimizer
{
  public:
	explicit differentiableFunctionMinimizer(char const *Name = "");

  private:
  public:
	//! Name of the differentiableFunctionMinimizer
	std::string name;

	// multi dimensional part
	TMFDMT *type;
	TMFD *fdf;

	T f;

	T *x;
	T *gradient;
	T *dx;

	std::size_t n;
};

#endif //UMUQ_FUNCTIONMINIMIZER_H
