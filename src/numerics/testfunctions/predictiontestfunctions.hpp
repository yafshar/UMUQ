#ifndef UMUQ_PREDICTIONTESTFUNCTIONS_H
#define UMUQ_PREDICTIONTESTFUNCTIONS_H

/*! \class qian
 * \ingroup Numerics_Module
 * 
 * \brief Qian's 1-Dimensional Function class
 * 
 * \tparam T data type
 * 
 * Reference:<br>
 * Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments: 
 * Test Functions and Datasets. Retrieved May 22, 2018, from http://www.sfu.ca/~ssurjano. 
 */
template <typename T>
struct qian
{
  /*!
   * \brief Construct a new qian object
   * 
   */
  qian() {}

  /*! 
   * \brief Qian's function
   * 
   * Qian's function
   * \f[
   * f(x) = e^{(3x)} \cos \left(\frac{7\pi x}{2}\right),~~ x \in [0 \cdots 1]  
   * \f]
   * 
   * \param  x  input data point
   * 
   * \returns f  function value at input data point
   */
  inline T f(T const *x)
  {
    return std::exp(3 * (*x)) * std::cos(3.5 * M_PI * (*x));
  }
};

/*! \class franke2d
 * \ingroup Numerics_Module
 * 
 * \brief Franke's 2-Dimensional Function class
 * 
 * \tparam T data type
 * 
 * Reference:<br>
 * Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments: 
 * Test Functions and Datasets. Retrieved May 22, 2018, from http://www.sfu.ca/~ssurjano. 
 */
template <typename T>
struct franke2d
{
  /*!
   * \brief Construct a new franke2d object
   * 
   */
  franke2d() {}

  /*!
   * \brief Franke's bivariate function
   * 
   * Franke's bivariate function is a weighted sum of four exponential
   * \f[
   * \begin{aligned} 
   * \nonumber f(x) &= 0.75 e^{\left(-\frac{(9x_1-2)^2}{4} - \frac{(9x_2-2)^2}{4} \right)} \\
   * \nonumber      &+ 0.75 e^{\left(-\frac{(9x_1+1)^2}{49} - \frac{(9x_2+1)}{10} \right)} \\
   * \nonumber      &+ 0.5 e^{\left(-\frac{(9x_1-7)^2}{4} - \frac{(9x_2-3)^2}{4} \right)} \\
   * \nonumber      &- 0.2 e^{\left(-(9x_1-4)^2 - (9x_2-7)^2 \right)}
   * \end{aligned}
   * \f]
   * 
   * \param  x  input data point
   * 
   * \returns f  function value at input data point
   */
  inline T f(T const *x)
  {
    T const x1 = x[0];
    T const x2 = x[1];
    T const t1 = 0.75 * std::exp(-std::pow(9 * x1 - 2, 2) / 4 - std::pow(9 * x2 - 2, 2) / 4);
    T const t2 = 0.75 * std::exp(-std::pow(9 * x1 + 1, 2) / 49 - (9 * x2 + 1) / 10);
    T const t3 = 0.5 * std::exp(-std::pow(9 * x1 - 7, 2) / 4 - std::pow(9 * x2 - 3, 2) / 4);
    T const t4 = -0.2 * std::exp(-std::pow(9 * x1 - 4, 2) - std::pow(9 * x2 - 7, 2));
    return t1 + t2 + t3 + t4;
  }
};

/*! \class rastrigin
 * \ingroup Numerics_Module
 * 
 * \brief Rastrigin's N-Dimensional Function class 
 * 
 * \tparam T data type
 * 
 * 
 * Reference:<br>
 * Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments: 
 * Test Functions and Datasets. Retrieved May 22, 2018, from http://www.sfu.ca/~ssurjano. 
 */
template <typename T>
class rastrigin
{
public:
  /*!
   * \brief Construct a new Rastrigin's function object
   * 
   * \param dim Dimension of the space (default is 2)
   */
  rastrigin(int dim = 2) : nDim(dim) {}

  /*! 
   * \brief Rastrigin function
   * 
   * The Rastrigin function has several local minima. 
   * It is highly multimodal, but locations of the minima are regularly distributed.
   * \f$ f(x) = 10d + \sum_{i=1}^{d} \left[x_i^2 -10 cos \left( 2 \pi x_i\right) \right] \f$
   * 
   * \param  x  input data point
   * 
   * \returns f  function value at input data point
   */
  inline T f(T const *x)
  {
    T sum(0);
    std::for_each(x, x + nDim, [&](T const i) { sum += i * i - 10 * std::cos(M_2PI * i); });
    return 10 * nDim + sum;
  }

private:
  //! Dimension of the problem
  int nDim;
};

/*! \class ackley
 * \ingroup Numerics_Module
 * 
 * \brief Ackley's N-Dimensional Function class 
 * 
 * \tparam T data type
 * 
 * Reference:<br>
 * Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments: 
 * Test Functions and Datasets. Retrieved May 22, 2018, from http://www.sfu.ca/~ssurjano. 
 */
template <typename T>
class ackley
{
public:
  /*!
   * \brief Construct a new Ackley's function object
   * 
   * \param dim Dimension of the space (default is 2)
   */
  ackley(int dim = 2) : nDim(dim), a(20), b(0.2), c(M_2PI) {}

  /*!
   * \brief Construct a new Ackley's function object
   * 
   * \param dim Dimension of the space (default is 2)
   * \param aa  A constant (optional), with default value 20
   * \param bb  A constant (optional), with default value 0.2
   * \param cc  A constant (optional), with default value \f$ 2*\pi \f$
   */
  ackley(int dim = 2, T aa = 20, T bb = 0.2, T cc = M_2PI) : nDim(dim), a(aa), b(bb), c(cc) {}

  /*!
   * \brief Ackley function
   * 
   * The Ackley function has several local minima, in 2-dimensional form it is characterized 
   * by a nearly flat outer region, and a large hole at the centre.
   * It poses a risk for optimization algorithms, to be trapped in one of its many local minima.
   * \f$ f(x) = -a e^{\left(-b \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2} \right)} - e^{\left(\frac{1}{d}\sum_{i=1}^{d} {\cos (cx_i)}\right)} + a + e^1 \f$
   * 
   * Recommended variable values are: \f$ a = 20, b = 0.2 \text{and} c = 2\pi . \f$ 
   * 
   * \param  x  input data point
   * 
   * \returns f  function value at input data point
   */
  inline T f(T const *x)
  {
    T sum1(0);
    T sum2(0);

    std::for_each(x, x + nDim, [&](T const i) { sum1 += i * i; });
    std::for_each(x, x + nDim, [&](T const i) { sum2 += std::cos(c * i); });

    T t1 = -a * std::exp(-b * std::sqrt(sum1 / nDim));
    T t2 = -std::exp(sum2 / nDim);
    return t1 + t2 + a + std::exp(1);
  }

private:
  //! Dimension of the problem
  int nDim;
  //! A constant (optional), with default value 20
  T a;
  //! A constant (optional), with default value 0.2
  T b;
  //! A constant (optional), with default value \f$ 2*\pi \f$
  T c;
};

/*! \class peaks
 * \ingroup Numerics_Module
 * 
 * \brief Matlab peaks 2-Dimensional Function class
 * 
 * Matlab peaks is a function of two variables, obtained by translating and scaling Gaussian distributions
 * 
 * \tparam T data type
 * 
 * Reference:<br>
 * https://www.mathworks.com/help/matlab/ref/peaks.html 
 */
template <typename T>
struct peaks
{
  /*!
   * \brief Construct a new qian object
   * 
   */
  peaks() {}

  /*!
   * \brief Matlab peaks's function
   * 
   * Matlab peaks's function <br>
   * \f$
   * f(x) = 3(1-x_1)^2e^{\left(-x_1^2-(x_2+1)^2\right)}-10\left(\frac{x_1}{5}-x_1^3-x_2^5\right)e^{\left(-x_1^2-x_2^2 \right)}-\frac{1}{3}e^{\left(-x_2^2-(x_1+1)^2\right)}
   * \f$
   * 
   * \param x  Input data point
   * 
   * \returns f Function value at input data point
   */
  inline T f(T const *x)
  {
    return 3 * std::pow(1 - x[0], 2) *
               std::exp(-std::pow(x[0], 2) - std::pow(x[1] + 1, 2)) -
           10 * (x[0] / 5. - std::pow(x[0], 3) - std::pow(x[1], 5)) *
               std::exp(-std::pow(x[0], 2) - std::pow(x[1], 2)) -
           1. / 3 * std::exp(-std::pow(x[0] + 1, 2) - std::pow(x[1], 2));
  }
};

#endif // UMUQ_PREDICTIONTESTFUNCTIONS
