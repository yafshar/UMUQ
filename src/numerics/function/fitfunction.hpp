#ifndef UMUQ_FITFUNCTION_H
#define UMUQ_FITFUNCTION_H

#include "functiontype.hpp"
#include "umuqfunction.hpp"

/*! \class fitFunction
 * \brief The base class for fit function which can be used in the inference process
 * 
 * \tparam T Data type
 * \tparam F Polymorphic function wrapper
 */
template <typename T, class F = F_FTYPE<T>>
class fitFunction
{
  public:
    /*!
     * \brief Construct a new fitFunction object
     * 
     * \param Name Multidimensional fitFunction name
     */
    explicit fitFunction(char const *Name = "");

    /*!
     * \brief Set the Function to be used in this fitFunction
     * 
     * \param umFun UMUQ Function to be used in this fitFunction
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    virtual bool set(umuqFunction<T, F> &umFun);

    /*!
     * \brief Set the Function to be used in this fitFunction
     * 
     * \param Fun Function to be used in this fitFunction
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    virtual bool set(F &Fun);
    virtual bool set(F const &Fun);

    /*!
     * \brief Initilize the fitFunction
     * 
     * \return true 
     * \return false 
     */
    virtual bool init();

    /*!
     * \brief Get the Name object
     * 
     * \return std::string const 
     */
    inline std::string const getName() const;

  private:
    // Make it noncopyable
    fitFunction(fitFunction<T, F> const &) = delete;

    // Make it not assignable
    fitFunction<T, F> &operator=(fitFunction<T, F> const &) = delete;

  public:
    //! Name of the fitFunction
    std::string name;

    //! Function to be used in this fitFunction
    umuqFunction<T, F> fun;
};

template <typename T, class F>
fitFunction<T, F>::fitFunction(char const *Name) : name(Name) {}

template <typename T, class F>
bool fitFunction<T, F>::set(umuqFunction<T, F> &umFun)
{
    if (umFun)
    {
        fun = std::move(umFun);
    }
    else
    {
        UMUQFAILRETURN("Function is not assigned!");
    }
    return true;
}

template <typename T, class F>
bool fitFunction<T, F>::set(F &Fun)
{
    if (Fun)
    {
        fun.f = Fun;
    }
    else
    {
        UMUQFAILRETURN("Function is not assigned!");
    }
    return true;
}

template <typename T, class F>
bool fitFunction<T, F>::set(F const &Fun)
{
    if (Fun)
    {
        fun.f = Fun;
    }
    else
    {
        UMUQFAILRETURN("Function is not assigned!");
    }
    return true;
}

template <typename T, class F>
bool fitFunction<T, F>::init()
{
    return true;
}

template <typename T, class F>
inline std::string const fitFunction<T, F>::getName() const
{
    return name;
}

#endif //UMUQ_FITFUNCTION
