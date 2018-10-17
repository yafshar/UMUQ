#ifndef UMUQ_FITFUNCTION_H
#define UMUQ_FITFUNCTION_H

#include "functiontype.hpp"
#include "umuqfunction.hpp"

namespace umuq
{

/*! \class fitFunction
 * \brief The base class for fit function which can be used in the inference process
 * 
 * \tparam T Data type
 * \tparam F Function type
 */
template <typename T, class F = FITFUN_T<T>>
class fitFunction : public umuqFunction<T, F>
{
  public:
    /*!
     * \brief Construct a new fitFunction object
     * 
     * \param Name Multidimensional fitFunction name
     */
    explicit fitFunction(char const *Name = "");

    /*!
     * \brief Construct a new fitFunction Function object
     * 
     * \param Params     Parameters of fitFunction object
     * \param NumParams  Number of parameters
     * \param Name       fitFunction name
     */
    fitFunction(T const *Params, int const NumParams, char const *Name = "");
    fitFunction(std::vector<T> const &Params, char const *Name = "");

    /*!
     * \brief Move constructor, Construct a new umuqFunction object
     * 
     * \param other umuqFunction object
     */
    fitFunction(fitFunction<T, F> &&other);

    /*!
     * \brief Move assignment operator
     * 
     */
    fitFunction<T, F> &operator=(fitFunction<T, F> &&other);

    /*!
     * \brief Destroy the fit Function object
     * 
     */
    ~fitFunction();

    /*!
     * \brief Set the Init Function object
     * 
     * \param InitFun Initialization function which has the fixed shape of bool() 
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    inline bool setInitFunction(std::function<bool()> &InitFun);
    inline bool setInitFunction(std::function<bool()> const &InitFun);

    /*!
     * \brief Set the fitting Function to be used
     * 
     * \param Fun  Fitting Function of type (class F)
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    inline bool setFitFunction(F &Fun);
    inline bool setFitFunction(F const &Fun);

    /*!
     * \brief Setting both the Init Function & fitting Function 
     * 
     * \param InitFun  Initialization function which has the fixed shape of bool()
     * \param Fun      Fitting Function of type (class F)
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    inline bool set(std::function<bool()> &InitFun, F &Fun);
    inline bool set(std::function<bool()> const &InitFun, F const &Fun);

    /*!
     * \brief Initialize the fitFunction
     * 
     * \return true 
     * \return false 
     */
    virtual bool init();

  protected:
    /*!
     * \brief Delete a fitFunction object copy construction
     * 
     * Make it noncopyable.
     */
    fitFunction(fitFunction<T, F> const &) = delete;

    /*!
     * \brief Delete a fitFunction object assignment
     * 
     * Make it nonassignable
     * 
     * \returns fitFunction<T, F>& 
     */
    fitFunction<T, F> &operator=(fitFunction<T, F> const &) = delete;

  public:
    //! Initialization function which has the type of bool
    std::function<bool()> initFun;
};

template <typename T, class F>
fitFunction<T, F>::fitFunction(char const *Name) : umuqFunction<T, F>(Name) {}

template <typename T, class F>
fitFunction<T, F>::fitFunction(T const *Params, int const NumParams, const char *Name) : umuqFunction<T, F>(Params, NumParams, Name) {}

template <typename T, class F>
fitFunction<T, F>::fitFunction(std::vector<T> const &Params, const char *Name) : umuqFunction<T, F>(Params, Name) {}

template <typename T, class F>
fitFunction<T, F>::fitFunction(fitFunction<T, F> &&other) : umuqFunction<T, F>::umuqFunction(std::move(other)),
                                                            initFun(std::move(other.initFun))
{
}

template <typename T, class F>
fitFunction<T, F> &fitFunction<T, F>::operator=(fitFunction<T, F> &&other)
{
    umuqFunction<T, F>::operator=(std::move(other));
    this->initFun = std::move(other.initFun);

    return *this;
}

template <typename T, class F>
fitFunction<T, F>::~fitFunction() {}

template <typename T, class F>
inline bool fitFunction<T, F>::setInitFunction(std::function<bool()> &InitFun)
{
    if (InitFun)
    {
        initFun = InitFun;
        return true;
    }
    UMUQFAILRETURN("Init function is not assigned!");
}

template <typename T, class F>
inline bool fitFunction<T, F>::setInitFunction(std::function<bool()> const &InitFun)
{
    if (InitFun)
    {
        initFun = InitFun;
        return true;
    }
    UMUQFAILRETURN("Init function is not assigned!");
}

template <typename T, class F>
inline bool fitFunction<T, F>::setFitFunction(F &Fun)
{
    if (Fun)
    {
        this->f = Fun;
        return true;
    }
    UMUQFAILRETURN("Fitting function is not assigned!");
}

template <typename T, class F>
inline bool fitFunction<T, F>::setFitFunction(F const &Fun)
{
    if (Fun)
    {
        this->f = Fun;
        return true;
    }
    UMUQFAILRETURN("Fitting function is not assigned!");
}

template <typename T, class F>
inline bool fitFunction<T, F>::set(std::function<bool()> &InitFun, F &Fun)
{
    if (InitFun)
    {
        initFun = InitFun;
        if (Fun)
        {
            this->f = Fun;
            return true;
        }
        UMUQWARNING("Fitting function is not assigned!");
        return true;
    }
    else
    {
        UMUQWARNING("Init function is not assigned!");
        if (Fun)
        {
            this->f = Fun;
            return true;
        }
        UMUQFAILRETURN("Fitting function is not assigned!");
    }
    UMUQFAILRETURN("Init function is not assigned!");
}

template <typename T, class F>
inline bool fitFunction<T, F>::set(std::function<bool()> const &InitFun, F const &Fun)
{
    if (InitFun)
    {
        initFun = InitFun;
        if (Fun)
        {
            this->f = Fun;
            return true;
        }
        UMUQWARNING("Fitting function is not assigned!");
        return true;
    }
    else
    {
        UMUQWARNING("Init function is not assigned!");
        if (Fun)
        {
            this->f = Fun;
            return true;
        }
        UMUQFAILRETURN("Fitting function is not assigned!");
    }
    UMUQFAILRETURN("Init function is not assigned!");
}

template <typename T, class F>
bool fitFunction<T, F>::init()
{
    if (initFun)
    {
        return initFun();
    }
    return true;
}

} // namespace umuq

#endif //UMUQ_FITFUNCTION
