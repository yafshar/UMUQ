#ifndef UMUQ_RESIDUAL_H
#define UMUQ_RESIDUAL_H

namespace umuq
{

/*! \enum ErrorTypes
 * \ingroup Numerics_Module
 * 
 * \brief Different residuals Error types, currently available in %UMUQ
 * 
 */
enum ErrorTypes
{
    /*! Absolute difference between observed and predicted data. */
    AbsoluteError = -1,
    /*! A ratio of absolute difference between observed and predicted data to the absolute value of observed data. */
    ScaledError = -2,
    /*! Squred value of the difference between observed and predicted data. */
    SquredError = -3
};

/*! \class residual
 * \ingroup Numerics_Module
 *
 * \brief Computes residuals of observation and predicted data based on different Error type
 *
 * List of available Error types:
 *  - \b AbsoluteError Absolute difference between observed and predicted data
 *  - \b ScaledError   It is a ratio of absolute difference between observed and predicted data  
 *                     to the absolute value of observed data
 *  - \b SquredError   Squred value of the difference between observed and predicted data
 */
template <typename T>
class residual
{
  public:
    /*!
     * \brief Construct a new residual object
     * 
     * \param ierrorType Input error type is a residual type (default AbsoluteError) 
     */
    residual(int const ierrorType = ErrorTypes::AbsoluteError);

    /*!
     * \brief Construct a new residual object
     * 
     * \param ierrorType Input error type is a residual type (default AbsoluteError) 
     */
    residual(std::string const &ierrorType);

    /*!
     * \brief set the new error type
     * 
     * \param  ierrorType  Input error type in computing residual
     * 
     * \return true 
     * \return false  if the error type is unknown 
     */
    bool set(std::string const &ierrorType);

    /*!
     * \brief set the new error type
     * 
     * \param  ierrorType  Input error type in computing residual
     * 
     * \return true 
     * \return false  if the error type is unknown 
     */
    bool set(int ierrorType);

    /*!
     * \brief Compute the residual based on error type
     * 
     * \param observed  Observed data
     * \param predicted Predicted data
     * 
     * \return Residual based on error type
     */
    inline T operator()(T const &observed, T const &predicted);

  private:
    // Make it noncopyable
    residual(residual<T> const &) = delete;

    // Make it not assignable
    residual<T> &operator=(residual<T> const &) = delete;

  private:
    //! Error type in computing residuals
    int errorType;
};


template <typename T>
residual<T>::residual(int const ierrorType) : errorType(ierrorType) {}

template <typename T>
residual<T>::residual(std::string const &ierrorType)
{
    if (ierrorType == "absolute" || ierrorType == "Absolute" || ierrorType == "AbsoluteError")
    {
        this->errorType = ErrorTypes::AbsoluteError;
    }
    else if (ierrorType == "scaled" || ierrorType == "Scaled" || ierrorType == "ScaledError")
    {
        this->errorType = ErrorTypes::ScaledError;
    }
    else if (ierrorType == "squared" || ierrorType == "Squared" || ierrorType == "SquredError")
    {
        this->errorType = ErrorTypes::SquredError;
    }
    else
    {
        UMUQWARNING("Error type is unknown : Change to the default absolute Error!");
        this->errorType = ErrorTypes::AbsoluteError;
    }
}

template <typename T>
bool residual<T>::set(std::string const &ierrorType)
{
    if (ierrorType == "absolute" || ierrorType == "Absolute" || ierrorType == "AbsoluteError")
    {
        this->errorType = ErrorTypes::AbsoluteError;
    }
    else if (ierrorType == "scaled" || ierrorType == "Scaled" || ierrorType == "ScaledError")
    {
        this->errorType = ErrorTypes::ScaledError;
    }
    else if (ierrorType == "squared" || ierrorType == "Squared" || ierrorType == "SquredError")
    {
        this->errorType = ErrorTypes::SquredError;
    }
    else
    {
        UMUQFAILRETURN("Error type is unknown!");
    }
    return true;
}

template <typename T>
bool residual<T>::set(int ierrorType)
{
    if (ierrorType == ErrorTypes::AbsoluteError || ierrorType == ErrorTypes::ScaledError || ierrorType == ErrorTypes::SquredError)
    {
        this->errorType = ierrorType;
    }
    else
    {
        UMUQFAILRETURN("Error type is unknown!");
    }
    return true;
}

template <typename T>
inline T residual<T>::operator()(T const &observed, T const &predicted)
{
    switch (this->errorType)
    {
    case ErrorTypes::AbsoluteError:
        return std::abs(observed - predicted);
    case ErrorTypes::ScaledError:
        return std::abs(observed - predicted) / std::abs(observed);
    case ErrorTypes::SquredError:
        return (observed - predicted) * (observed - predicted);
    default:
        return std::abs(observed - predicted);
    }
}

} // namespace umuq

#endif //UMUQ_RESIDUAL
