#ifndef UMUQ_DENSITYFUNCTION_H
#define UMUQ_DENSITYFUNCTION_H

#include "../../core/core.hpp"

/*! \class densityFunction
 * \brief Density function class 
 * CRTP pattern
 * 
 * \tparam T   Data type
 * \tparam TD  Inheriting from a template class, where the derived class itself as a template parameter of the base class
 */
template <typename T, class TD>
class densityFunction
{
  public:
    /*!
     * \brief Construct a new density Function object
     * 
     */
    densityFunction() : name(""), numParams(0) {}

    /*!
     * \brief Construct a new density Function object
     * 
     * \param NumParams  Number of parameters
     */
    densityFunction(T *Params, int const NumParams, const char *Name = "") : name(Name),
                                                                             numParams(NumParams),
                                                                             params(Params, Params + NumParams)
    {
    }

    /*!
     * \brief reset the number of parameters and its argument
     * 
     * \param NumParams Number of parameters 
     * 
     * \return true 
     * \return false 
     */
    bool reset(T *Params, int const NumParams, const char *Name = "")
    {
        numParams = NumParams;
        try
        {
            params.resize(numParams);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to resize memory!")
        }
        std::copy(Params, Params + NumParams, params.data());

        name = std::string(Name);
    }

    /*!
     * \brief Density function
     * 
     * \param x Input value
     * 
     * \returns Density function value 
     */
    inline T f(T const x)
    {
        return static_cast<TD *>(this)->f(x);
    }

    /*!
     * \brief Log of density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T lf(T const x)
    {
        return static_cast<TD *>(this)->lf(x);
    }

  public:
    // Name of the function
    std::string name;

    // Number of parameters
    int numParams;

    // Denisty function parameters
    std::vector<T> params;

  private:
    friend TD;
};

/*! \class uniformDistribution
 * \brief Flat (Uniform) Distribution
 * 
 * This class provides probability density p(x) at x for a uniform distribution from \f$ \[a \cdots b\] \f$, 
 * using: 
 * \f[
 * p(x)= \left\{\begin{matrix}
 * 1/(b-a)  &a \leqslant  x < b \\ 
 *  0       &otherwise
 * \end{matrix}\right.
 * \f]
 * 
 * \tparam T Data type
 */
template <typename T>
class uniformDistribution : public densityFunction<T, uniformDistribution<T>>
{
  public:
    uniformDistribution(T const a, T const b) : densityFunction<T, uniformDistribution<T>>(std::vector<T>{a, b}.data(), 2, "uniform") {}

    ~uniformDistribution() {}

    /*!
     * \brief Density function
     * 
     * \param x Input value
     * 
     * \returns Density function value 
     */
    inline T f(T const x)
    {
        return (x < this->params[1] && x >= this->params[0]) ? static_cast<T>(1) / (this->params[1] - this->params[0]) : T{};
    }

    /*!
     * \brief Log of density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T lf(T const x)
    {
        return (x < this->params[1] && x >= this->params[0]) ? -std::log(this->params[1] - this->params[0]) : std::numeric_limits<T>::infinity();
    }
};

/*! \class gaussianDistribution
 * \brief The Gaussian Distribution
 * 
 * This class provides probability density \f$ p(x) \f$ at x for a uniform distribution from \f$ \[a \cdots b\] \f$, 
 * using: 
 * \f[
 * p(x)= \left\{\begin{matrix}
 * 1/(b-a)  &a \leqslant  x < b \\ 
 *  0       &otherwise
 * \end{matrix}\right.
 * \f]
 * 
 * \tparam T Data type
 */
template <typename T>
class gaussianDistribution : public densityFunction<T, uniformDistribution<T>>
{

};




#endif // UMUQ_DENSITYFUNCTION_H