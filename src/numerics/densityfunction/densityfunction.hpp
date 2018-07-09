#ifndef UMUQ_DENSITYFUNCTION_H
#define UMUQ_DENSITYFUNCTION_H

#include "../factorial.hpp"
#include "../eigenlib.hpp"
#include "../random/psrandom.hpp"

/*! \class densityFunction
 * \brief Density function class 
 * CRTP pattern
 * 
 * A density function or a probability density (PDF), is a function, with a value at any given point (or sample point) 
 * interpreted as a relative likelihood that the value of the random variable would be equal to that sample.
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
    densityFunction();

    /*!
     * \brief Construct a new density Function object
     * 
     * \param Params     Parameters of density Function object
     * \param NumParams  Number of parameters
     * \param Name       Distribution name
     */
    densityFunction(T const *Params, int const NumParams, const char *Name = "");

    /*!
     * \brief Destroy the density Function object
     * 
     */
    ~densityFunction() {}

    /*!
     * \brief reset the number of parameters and its argument
     * 
     * \param NumParams Number of parameters 
     * 
     * \return true 
     * \return false 
     */
    bool reset(T const *Params, int const NumParams, const char *Name = "");

    /*!
     * \brief Density function
     * 
     * \param x Input value
     * 
     * \returns Density function value 
     */
    inline T f(T const x);

    inline T f(T const *x);

    inline T f(EVectorX<T> const &x);

    template <typename Y>
    inline T f(T const *x, Y const *y);

    /*!
     * \brief Log of density function
     * 
     * \param x Input value
     * 
     * \returns  Log of density function value 
     */
    inline T lf(T const x);

    inline T lf(T const *x);

    inline T lf(EVectorX<T> const &x);

    template <typename Y>
    inline T lf(T const *x, Y const *y);

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

/*!
 * \brief Construct a new density Function object
 * 
 */
template <typename T, class TD>
densityFunction<T, TD>::densityFunction() : name(""), numParams(0) {}

/*!
 * \brief Construct a new density Function object
 * 
 * \param Params     Parameters of density Function object
 * \param NumParams  Number of parameters
 * \param Name       Distribution name
 */
template <typename T, class TD>
densityFunction<T, TD>::densityFunction(T const *Params, int const NumParams, const char *Name) : name(Name),
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
template <typename T, class TD>
bool densityFunction<T, TD>::reset(T const *Params, int const NumParams, const char *Name)
{
    this->name = std::string(Name);
    this->numParams = NumParams;
    try
    {
        this->params.resize(NumParams);
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to resize memory!")
    }
    std::copy(Params, Params + NumParams, this->params.data());
}

/*!
 * \brief Density function
 * 
 * \param x Input value
 * 
 * \returns Density function value 
 */
template <typename T, class TD>
inline T densityFunction<T, TD>::f(T const x)
{
    return static_cast<TD *>(this)->f(x);
}

template <typename T, class TD>
inline T densityFunction<T, TD>::f(T const *x)
{
    return static_cast<TD *>(this)->f(x);
}

template <typename T, class TD>
inline T densityFunction<T, TD>::f(EVectorX<T> const &x)
{
    return static_cast<TD *>(this)->f(x);
}

template <typename T, class TD>
template <typename Y>
inline T densityFunction<T, TD>::f(T const *x, Y const *y)
{
    return static_cast<TD *>(this)->template f<Y>(x, y);
}

/*!
 * \brief Log of density function
 * 
 * \param x Input value
 * 
 * \returns  Log of density function value 
 */
template <typename T, class TD>
inline T densityFunction<T, TD>::lf(T const x)
{
    return static_cast<TD *>(this)->lf(x);
}

template <typename T, class TD>
inline T densityFunction<T, TD>::lf(T const *x)
{
    return static_cast<TD *>(this)->lf(x);
}

template <typename T, class TD>
inline T densityFunction<T, TD>::lf(EVectorX<T> const &x)
{
    return static_cast<TD *>(this)->lf(x);
}

template <typename T, class TD>
template <typename Y>
inline T densityFunction<T, TD>::lf(T const *x, Y const *y)
{
    return static_cast<TD *>(this)->template lf<Y>(x, y);
}

#endif // UMUQ_DENSITYFUNCTION_H
