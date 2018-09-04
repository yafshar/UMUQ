#ifndef UMUQ_FITFUNCTION_H
#define UMUQ_FITFUNCTION_H

#include "functiontype.hpp"
#include "umuqfunction.hpp"

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
     * \brief Destroy the fit Function object
     * 
     */
    ~fitFunction();

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
     * \brief Set the Init Function object
     * 
     * \param InitFun Initilization function which has the fixed shape of bool() 
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    bool setInitFunction(std::function<bool()> &InitFun);
    bool setInitFunction(std::function<bool()> const &InitFun);

    /*!
     * \brief Set the fitting Function to be used
     * 
     * \param fitFun  Fitting Function of type (class F)
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    bool setfitFunction(F &fitFun);
    bool setfitFunction(F const &fitFun);

    /*!
     * \brief Setting both the Init Function & fitting Function 
     * 
     * \param InitFun  Initilization function which has the fixed shape of bool()
     * \param fitFun   Fitting Function of type (class F)
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    bool set(std::function<bool()> &InitFun, F &fitFun);
    bool set(std::function<bool()> const &InitFun, F const &fitFun);

    /*!
     * \brief Initilize the fitFunction
     * 
     * \return true 
     * \return false 
     */
    virtual bool init();

  private:
    // Make it noncopyable
    fitFunction(fitFunction<T, F> const &) = delete;

    // Make it not assignable
    fitFunction<T, F> &operator=(fitFunction<T, F> const &) = delete;

  public:
    //! Initilization function which has the fixed shape of bool()
    std::function<bool()> initFun;
};

template <typename T, class F>
fitFunction<T, F>::fitFunction(char const *Name) : umuqFunction<T, F>(Name) {}

template <typename T, class F>
fitFunction<T, F>::fitFunction(T const *Params, int const NumParams, const char *Name) : umuqFunction<T, F>(Params, NumParams, Name) {}

template <typename T, class F>
fitFunction<T, F>::fitFunction(std::vector<T> const &Params, const char *Name) : umuqFunction<T, F>(Params, Name) {}

template <typename T, class F>
fitFunction<T, F>::~fitFunction() {}

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
bool fitFunction<T, F>::setInitFunction(std::function<bool()> &InitFun)
{
    if (InitFun)
    {
        initFun = InitFun;
        return true;
    }
    UMUQFAILRETURN("Init function is not assigned!");
}

template <typename T, class F>
bool fitFunction<T, F>::setInitFunction(std::function<bool()> const &InitFun)
{
    if (InitFun)
    {
        initFun = InitFun;
        return true;
    }
    UMUQFAILRETURN("Init function is not assigned!");
}

template <typename T, class F>
bool fitFunction<T, F>::setfitFunction(F &fitFun)
{
    if (fitFun)
    {
        this->f = fitFun;
        return true;
    }
    UMUQFAILRETURN("Fitting function is not assigned!");
}

template <typename T, class F>
bool fitFunction<T, F>::setfitFunction(F const &fitFun)
{
    if (fitFun)
    {
        this->f = fitFun;
        return true;
    }
    UMUQFAILRETURN("Fitting function is not assigned!");
}

template <typename T, class F>
bool fitFunction<T, F>::set(std::function<bool()> &InitFun, F &fitFun)
{
    if (InitFun)
    {
        initFun = InitFun;
        if (fitFun)
        {
            this->f = fitFun;
            return true;
        }
        UMUQWARNING("Fitting function is not assigned!");
        return true;
    }
    else
    {
        UMUQWARNING("Init function is not assigned!");
        if (fitFun)
        {
            this->f = fitFun;
            return true;
        }
        UMUQFAILRETURN("Fitting function is not assigned!");
    }
    UMUQFAILRETURN("Init function is not assigned!");
}

template <typename T, class F>
bool fitFunction<T, F>::set(std::function<bool()> const &InitFun, F const &fitFun)
{
    if (InitFun)
    {
        initFun = InitFun;
        if (fitFun)
        {
            this->f = fitFun;
            return true;
        }
        UMUQWARNING("Fitting function is not assigned!");
        return true;
    }
    else
    {
        UMUQWARNING("Init function is not assigned!");
        if (fitFun)
        {
            this->f = fitFun;
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

#endif //UMUQ_FITFUNCTION
