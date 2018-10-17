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
    /*! Squared value of the difference between observed and predicted data. */
    SquaredError = -3
};

/*! \class residual
 * \ingroup Numerics_Module
 *
 * \brief Computes residuals of observation and predicted data based on different Error type
 *
 * List of available Error types:<br>
 *  - \b AbsoluteError Absolute difference between observed and predicted data
 *  - \b ScaledError   It is a ratio of absolute difference between observed and predicted data  
 *                     to the absolute value of observed data
 *  - \b SquaredError  Squared value of the difference between observed and predicted data
 */
template <typename T>
class residual
{
  public:
    /*!
     * \brief Construct a new residual object
     * 
     * \param ErrorType Input error type is a residual type (default AbsoluteError) 
     */
    residual(int const ErrorType = ErrorTypes::AbsoluteError);

    /*!
     * \brief Construct a new residual object
     * 
     * \param ErrorType Input error type is a residual type (default AbsoluteError) 
     */
    residual(std::string const &ErrorType);

    /*!
     * \brief set the new error type
     * 
     * \param  ErrorType  Input error type in computing residual
     * 
     * \return false  if the error type is unknown 
     */
    bool set(std::string const &ErrorType);

    /*!
     * \brief set the new error type
     * 
     * \param  ErrorType  Input error type in computing residual
     * 
     * \return false  if the error type is unknown 
     */
    bool set(int ErrorType);

    /*!
     * \brief Compute the residual based on error type
     * 
     * \param observed  Observed data
     * \param predicted Predicted data
     * 
     * \return Residual based on error type
     */
    inline T operator()(T const &observed, T const &predicted);

  protected:
    /*!
     * \brief Delete a residual object copy construction
     * 
     * Make it noncopyable.
     */
    residual(residual<T> const &) = delete;

    /*!
     * \brief Delete a residual object assignment
     * 
     * Make it nonassignable
     * 
     * \returns residual<T>& 
     */
    residual<T> &operator=(residual<T> const &) = delete;

  private:
    //! Error type in computing residuals
    int errorType;
};

template <typename T>
residual<T>::residual(int const ErrorType) : errorType(ErrorType) {}

template <typename T>
residual<T>::residual(std::string const &ErrorType)
{
    if (ErrorType == "absolute" || ErrorType == "Absolute" || ErrorType == "AbsoluteError")
    {
        errorType = ErrorTypes::AbsoluteError;
    }
    else if (ErrorType == "scaled" || ErrorType == "Scaled" || ErrorType == "ScaledError")
    {
        errorType = ErrorTypes::ScaledError;
    }
    else if (ErrorType == "squared" || ErrorType == "Squared" || ErrorType == "SquaredError")
    {
        errorType = ErrorTypes::SquaredError;
    }
    else
    {
        UMUQWARNING("Error type is unknown : Change to the default absolute Error!");

        errorType = ErrorTypes::AbsoluteError;
    }
}

template <typename T>
bool residual<T>::set(std::string const &ErrorType)
{
    if (ErrorType == "absolute" || ErrorType == "Absolute" || ErrorType == "AbsoluteError")
    {
        errorType = ErrorTypes::AbsoluteError;
    }
    else if (ErrorType == "scaled" || ErrorType == "Scaled" || ErrorType == "ScaledError")
    {
        errorType = ErrorTypes::ScaledError;
    }
    else if (ErrorType == "squared" || ErrorType == "Squared" || ErrorType == "SquaredError")
    {
        errorType = ErrorTypes::SquaredError;
    }
    else
    {
        UMUQFAILRETURN("Error type is unknown!");
    }
    return true;
}

template <typename T>
bool residual<T>::set(int ErrorType)
{
    if (ErrorType == ErrorTypes::AbsoluteError || ErrorType == ErrorTypes::ScaledError || ErrorType == ErrorTypes::SquaredError)
    {
        errorType = ErrorType;
        return true;
    }
    else
    {
        UMUQFAILRETURN("Error type is unknown!");
    }
}

template <typename T>
inline T residual<T>::operator()(T const &observed, T const &predicted)
{
    switch (errorType)
    {
    case ErrorTypes::AbsoluteError:
        return std::abs(observed - predicted);
    case ErrorTypes::ScaledError:
        return std::abs(observed - predicted) / std::abs(observed);
    case ErrorTypes::SquaredError:
        return (observed - predicted) * (observed - predicted);
    default:
        return std::abs(observed - predicted);
    }
}

} // namespace umuq

#endif //UMUQ_RESIDUAL
