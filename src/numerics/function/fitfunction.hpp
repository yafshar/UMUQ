#ifndef UMUQ_FITFUNCTION_H
#define UMUQ_FITFUNCTION_H

#include "datatype/functiontype.hpp"
#include "umuqfunction.hpp"

namespace umuq
{

/*! \class fitFunction
 * \brief The base class for fit function which can be used in the inference process
 * 
 * \tparam DataType     Data type
 * \tparam FunctionType Function type
 */
template <typename DataType, class FunctionType = FITFUN_T<DataType>>
class fitFunction : public umuqFunction<DataType, FunctionType>
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
    fitFunction(DataType const *Params, int const NumParams, char const *Name = "");

    /*!
     * \brief Construct a new fitFunction Function object
     * 
     * \param Params     Parameters of fitFunction object
     * \param Name       fitFunction name
     */
    fitFunction(std::vector<DataType> const &Params, char const *Name = "");

    /*!
     * \brief Move constructor, Construct a new umuqFunction object
     * 
     * \param other umuqFunction object
     */
    fitFunction(fitFunction<DataType, FunctionType> &&other);

    /*!
     * \brief Move assignment operator
     * 
     */
    fitFunction<DataType, FunctionType> &operator=(fitFunction<DataType, FunctionType> &&other);

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
     * \return false If it encounters an unexpected problem
     */
    inline bool setInitFunction(std::function<bool()> &InitFun);

    /*!
     * \brief Set the Init Function object
     * 
     * \param InitFun Initialization function which has the fixed shape of bool() 
     * 
     * \return false If it encounters an unexpected problem
     */
    inline bool setInitFunction(std::function<bool()> const &InitFun);

    /*!
     * \brief Set the fitting Function to be used
     * 
     * \param Fun  Fitting Function of type (class FunctionType)
     * 
     * \return false If it encounters an unexpected problem
     */
    inline bool setFitFunction(FunctionType &Fun);

    /*!
     * \brief Set the fitting Function to be used
     * 
     * \param Fun  Fitting Function of type (class FunctionType)
     * 
     * \return false If it encounters an unexpected problem
     */
    inline bool setFitFunction(FunctionType const &Fun);

    /*!
     * \brief Setting both the Init Function & fitting Function 
     * 
     * \param InitFun  Initialization function which has the fixed shape of bool()
     * \param Fun      Fitting Function of type (class FunctionType)
     * 
     * \return false If it encounters an unexpected problem
     */
    inline bool set(std::function<bool()> &InitFun, FunctionType &Fun);

    /*!
     * \brief Setting both the Init Function & fitting Function 
     * 
     * \param InitFun  Initialization function which has the fixed shape of bool()
     * \param Fun      Fitting Function of type (class FunctionType)
     * 
     * \return false If it encounters an unexpected problem
     */
    inline bool set(std::function<bool()> const &InitFun, FunctionType const &Fun);

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
     * Avoiding implicit generation of the copy constructor.
     */
    fitFunction(fitFunction<DataType, FunctionType> const &) = delete;

    /*!
     * \brief Delete a fitFunction object assignment
     * 
     * Avoiding implicit copy assignment.
     * 
     * \returns fitFunction<DataType, FunctionType>& 
     */
    fitFunction<DataType, FunctionType> &operator=(fitFunction<DataType, FunctionType> const &) = delete;

  public:
    /*! Initialization function which has the type of bool */
    std::function<bool()> initFun;
};

template <typename DataType, class FunctionType>
fitFunction<DataType, FunctionType>::fitFunction(char const *Name) : umuqFunction<DataType, FunctionType>(Name) {}

template <typename DataType, class FunctionType>
fitFunction<DataType, FunctionType>::fitFunction(DataType const *Params, int const NumParams, const char *Name) : umuqFunction<DataType, FunctionType>(Params, NumParams, Name) {}

template <typename DataType, class FunctionType>
fitFunction<DataType, FunctionType>::fitFunction(std::vector<DataType> const &Params, const char *Name) : umuqFunction<DataType, FunctionType>(Params, Name) {}

template <typename DataType, class FunctionType>
fitFunction<DataType, FunctionType>::fitFunction(fitFunction<DataType, FunctionType> &&other) : umuqFunction<DataType, FunctionType>::umuqFunction(std::move(other)),
                                                                                                initFun(std::move(other.initFun))
{
}

template <typename DataType, class FunctionType>
fitFunction<DataType, FunctionType> &fitFunction<DataType, FunctionType>::operator=(fitFunction<DataType, FunctionType> &&other)
{
    umuqFunction<DataType, FunctionType>::operator=(std::move(other));
    this->initFun = std::move(other.initFun);

    return *this;
}

template <typename DataType, class FunctionType>
fitFunction<DataType, FunctionType>::~fitFunction() {}

template <typename DataType, class FunctionType>
inline bool fitFunction<DataType, FunctionType>::setInitFunction(std::function<bool()> &InitFun)
{
    if (InitFun)
    {
        initFun = InitFun;
        return true;
    }
    UMUQFAILRETURN("Init function is not assigned!");
}

template <typename DataType, class FunctionType>
inline bool fitFunction<DataType, FunctionType>::setInitFunction(std::function<bool()> const &InitFun)
{
    if (InitFun)
    {
        initFun = InitFun;
        return true;
    }
    UMUQFAILRETURN("Init function is not assigned!");
}

template <typename DataType, class FunctionType>
inline bool fitFunction<DataType, FunctionType>::setFitFunction(FunctionType &Fun)
{
    if (Fun)
    {
        this->f = Fun;
        return true;
    }
    UMUQFAILRETURN("Fitting function is not assigned!");
}

template <typename DataType, class FunctionType>
inline bool fitFunction<DataType, FunctionType>::setFitFunction(FunctionType const &Fun)
{
    if (Fun)
    {
        this->f = Fun;
        return true;
    }
    UMUQFAILRETURN("Fitting function is not assigned!");
}

template <typename DataType, class FunctionType>
inline bool fitFunction<DataType, FunctionType>::set(std::function<bool()> &InitFun, FunctionType &Fun)
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

template <typename DataType, class FunctionType>
inline bool fitFunction<DataType, FunctionType>::set(std::function<bool()> const &InitFun, FunctionType const &Fun)
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

template <typename DataType, class FunctionType>
bool fitFunction<DataType, FunctionType>::init()
{
    if (initFun)
    {
        return initFun();
    }
    return true;
}

} // namespace umuq

#endif //UMUQ_FITFUNCTION
