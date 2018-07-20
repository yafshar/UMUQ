#ifndef UMUQ_FUNCTION_H
#define UMUQ_FUNCTION_H

#include "eigenlib.hpp"

/*! \class umuqfunction
 * \brief umuqfunction is a general-purpose polymorphic function wrapper of n variables
 * 
 * The Curiously Recurring Template Pattern (CRTP) for general-purpose polymorphic function wrapper of n variables
 * This class does not satisfy the requirements of CopyConstructible and CopyAssignable
 * 
 * \sa class densityFunction
 * \sa class multimin_function
 * 
 * \tparam T   Data type
 * \tparam TF  Function type (umuqfunction is inheriting from a template class)
 *             use the derived class itself as a template parameter of the base class
 */
template <typename T, class TF>
class umuqfunction
{
  public:
    /*!
     * \brief Construct a new umuqfunction object
     * 
     * \param Name  Function name
     */
    umuqfunction(char const Name = "");

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
     * \brief Rset the name, and number of dimension of the managed object
     * 
     * \param nDim   
     * \param Name 
     * \return true 
     * \return false 
     */
    inline bool reset(int const nDim, char const *Name = "");
    inline bool reset(T const *Params, int const NumParams, char const *Name = "");

    inline std::string name();

    inline int size();

    inline T *data();

    inline T *get();

    inline T f(T const x);
    inline T f(T const *x);
    template <typename X>
    inline T f(X const &x);
    template <typename Y>
    inline T f(T const x, Y const y);
    template <typename Y>
    inline T f(T const *x, Y const *y);
    template <typename X, typename Y>
    inline T f(X const &x, Y const &y);

    inline T df(T const x);
    inline T df(T const *x);
    template <typename X>
    inline T df(X const &x);
    template <typename Y>
    inline T df(T const x, Y const y);
    template <typename Y>
    inline T df(T const *x, Y const *y);
    template <typename X, typename Y>
    inline T df(X const &x, Y const &y);

    inline T lf(T const x);
    inline T lf(T const *x);
    template <typename X>
    inline T lf(X const &x);
    template <typename Y>
    inline T lf(T const x, Y const y);
    template <typename Y>
    inline T lf(T const *x, Y const *y);
    template <typename X, typename Y>
    inline T lf(X const &x, Y const &y);

    inline void fdf(T const x, T &func, T &dfunc);
    inline void fdf(T const *x, T &func, T &dfunc);
    template <typename Y>
    inline void fdf(Y const &y, T &func, T &dfunc);
    template <typename Y>
    inline void fdf(T const x, Y const y, T &func, T &dfunc);
    template <typename Y>
    inline void fdf(T const *x, Y const *y, T &func, T &dfunc);
    template <typename X, typename Y>
    inline void fdf(X const &x, Y const &y, T &func, T &dfunc);

  private:
    // Make it noncopyable
    umuqfunction(umuqfunction<T, TF> const &) = delete;

    // Make it not assignable
    umuqfunction<T, TF> &operator=(umuqfunction<T, TF> const &) = delete;

  private:
    // Name of the function
    std::string functionName;

    // Number of dimensions
    std::size_t numberofDimensions;

    // Function parameters
    std::vector<T> functionParameters;

  private:
    friend TF;
};

template <typename T, class TF>
umuqfunction<T, TF>::umuqfunction(char const Name) : functionName(Name),
                                                     numberofDimensions(0)
{
}
template <typename T, class TF>
umuqfunction<T, TF>::umuqfunction(int const nDim, char const *Name = "") : functionName(Name),
                                                                           numberofDimensions(nDim > 0 ? nDim : 0)
{
}
template <typename T, class TF>
umuqfunction<T, TF>::umuqfunction(T const *Params, int const NumParams, char const *Name = "") : functionName(Name),
                                                                                                 numberofDimensions(NumParams > 0 ? NumParams : 0),
                                                                                                 functionParameters(Params, Params + NumParams)
{
}

template <typename T, class TF>
void umuqfunction<T, TF>::reset(int const nDim, char const *Name = "")
{
    this->functionName = std::string(Name);
    this->numberofDimensions = nDim > 0 ? nDim : 0;
    return true;
}

template <typename T, class TF>
void umuqfunction<T, TF>::reset(T const *Params, int const NumParams, char const *Name = "")
{

    this->functionName = std::string(Name);
    this->numberofDimensions = NumParams > 0 ? NumParams : 0;
    try
    {
        this->functionParameters.resize(this->numberofDimensions);
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to resize memory!")
    }
    std::copy(Params, Params + NumParams, this->functionParameters.data());
    return true;
}

template <typename T, class TF>
inline std::string umuqfunction<T, TF>::name()
{
    return this->functionName;
}

template <typename T, class TF>
inline int umuqfunction<T, TF>::size()
{
    return static_cast<int>(this->numberofDimensions);
}

template <typename T, class TF>
inline T *umuqfunction<T, TF>::data()
{
    return this->functionParameters.data();
}

template <typename T, class TF>
inline T *umuqfunction<T, TF>::get()
{
    return this->functionParameters.data();
}

template <typename T, class TF>
inline T umuqfunction<T, TF>::f(T const x)
{
    return static_cast<T *>(this)->f(x);
}

template <typename T, class TF>
inline T umuqfunction<T, TF>::f(T const *x)
{
    return static_cast<T *>(this)->f(x);
}

template <typename T, class TF>
template <typename X>
inline T umuqfunction<T, TF>::f(X const &x)
{
    return static_cast<T *>(this)->f(y);
}

template <typename T, class TF>
template <typename Y>
inline T umuqfunction<T, TF>::f(T const x, Y const y)
{
    return static_cast<T *>(this)->f(x, y);
}

template <typename T, class TF>
template <typename Y>
inline T umuqfunction<T, TF>::f(T const *x, Y const *y)
{
    return static_cast<T *>(this)->f(x, y);
}

template <typename T, class TF>
template <typename X, typename Y>
inline T umuqfunction<T, TF>::f(X const &x, Y const &y)
{
    return static_cast<T *>(this)->f(x, y);
}

template <typename T, class TF>
inline T umuqfunction<T, TF>::df(T const x)
{
    return static_cast<T *>(this)->df(x);
}

template <typename T, class TF>
inline T umuqfunction<T, TF>::df(T const *x)
{
    return static_cast<T *>(this)->df(x);
}

template <typename T, class TF>
template <typename X>
inline T umuqfunction<T, TF>::df(X const &x)
{
    return static_cast<T *>(this)->df(y);
}

template <typename T, class TF>
template <typename Y>
inline T umuqfunction<T, TF>::df(T const x, Y const y)
{
    return static_cast<T *>(this)->df(x, y);
}

template <typename T, class TF>
template <typename Y>
inline T umuqfunction<T, TF>::df(T const *x, Y const *y)
{
    return static_cast<T *>(this)->df(x, y);
}

template <typename T, class TF>
template <typename X, typename Y>
inline T umuqfunction<T, TF>::df(X const &x, Y const &y)
{
    return static_cast<T *>(this)->df(x, y);
}

template <typename T, class TF>
inline T umuqfunction<T, TF>::lf(T const x)
{
    return static_cast<T *>(this)->lf(x);
}

template <typename T, class TF>
inline T umuqfunction<T, TF>::lf(T const *x)
{
    return static_cast<T *>(this)->lf(x);
}

template <typename T, class TF>
template <typename X>
inline T umuqfunction<T, TF>::lf(X const &x)
{
    return static_cast<T *>(this)->lf(y);
}

template <typename T, class TF>
template <typename Y>
inline T umuqfunction<T, TF>::lf(T const x, Y const y)
{
    return static_cast<T *>(this)->lf(x, y);
}

template <typename T, class TF>
template <typename Y>
inline T umuqfunction<T, TF>::lf(T const *x, Y const *y)
{
    return static_cast<T *>(this)->lf(x, y);
}

template <typename T, class TF>
template <typename X, typename Y>
inline T umuqfunction<T, TF>::lf(X const &x, Y const &y)
{
    return static_cast<T *>(this)->lf(x, y);
}

template <typename T, class TF>
inline void umuqfunction<T, TF>::fdf(T const x, T &func, T &dfunc)
{
    static_cast<T *>(this)->fdf(x, func, dfunc);
}

template <typename T, class TF>
inline void umuqfunction<T, TF>::fdf(T const *x, T &func, T &dfunc)
{
    static_cast<T *>(this)->fdf(x, func, dfunc);
}

template <typename T, class TF>
template <typename X>
inline void umuqfunction<T, TF>::fdf(X const &x, T &func, T &dfunc)
{
    static_cast<T *>(this)->fdf(x, func, dfunc);
}

template <typename T, class TF>
template <typename Y>
inline void umuqfunction<T, TF>::fdf(T const x, Y const y, T &func, T &dfunc)
{
    static_cast<T *>(this)->fdf(x, y, func, dfunc);
}

template <typename T, class TF>
template <typename Y>
inline void umuqfunction<T, TF>::fdf(T const *x, Y const *y, T &func, T &dfunc)
{
    static_cast<T *>(this)->fdf(x, y, func, dfunc);
}

template <typename T, class TF>
template <typename X, typename Y>
inline void umuqfunction<T, TF>::fdf(X const &x, Y const &y, T &func, T &dfunc)
{
    static_cast<T *>(this)->fdf(x, y, func, dfunc);
}

#endif // UMUQ_FUNCTION_H
