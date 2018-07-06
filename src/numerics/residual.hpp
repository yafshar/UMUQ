#ifndef UMUQ_RESIDUAL_H
#define UMUQ_RESIDUAL_H

enum
{
    AbsoluteError = -1,
    ScaledError = -2,
    SquredError = -3
};

/*! \class residual
 * \ingroup numerics
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
    residual(int const ierrorType = AbsoluteError);

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
    T operator()(T const &observed, T const &predicted);

  private:
    //! Error type in computing residuals
    int errorType;
};

/*!
 * \brief Construct a new residual object
 * 
 * \param ierrorType Input error type is a residual type (default AbsoluteError) 
 */
template <typename T>
residual<T>::residual(int const ierrorType) : errorType(ierrorType) {}

/*!
 * \brief Construct a new residual object
 * 
 * \param ierrorType Input error type is a residual type (default AbsoluteError)
 * 
 */
template <typename T>
residual<T>::residual(std::string const &ierrorType)
{
    if (ierrorType == "absolute" || ierrorType == "Absolute" || ierrorType == "AbsoluteError")
    {
        this->errorType = AbsoluteError;
    }
    else if (ierrorType == "scaled" || ierrorType == "Scaled" || ierrorType == "ScaledError")
    {
        this->errorType = ScaledError;
    }
    else if (ierrorType == "squared" || ierrorType == "Squared" || ierrorType == "SquredError")
    {
        this->errorType = SquredError;
    }
    else
    {
        UMUQWARNING("Error type is unknown : Change to the default absolute Error!");
        this->errorType = AbsoluteError;
    }
}

/*!
 * \brief set the new error type
 * 
 * \param  ierrorType  Input error type in computing residual
 * 
 * \return true
 * \return false  if the error type is unknown
 * 
 */
template <typename T>
bool residual<T>::set(std::string const &ierrorType)
{
    if (ierrorType == "absolute" || ierrorType == "Absolute" || ierrorType == "AbsoluteError")
    {
        this->errorType = AbsoluteError;
    }
    else if (ierrorType == "scaled" || ierrorType == "Scaled" || ierrorType == "ScaledError")
    {
        this->errorType = ScaledError;
    }
    else if (ierrorType == "squared" || ierrorType == "Squared" || ierrorType == "SquredError")
    {
        this->errorType = SquredError;
    }
    else
    {
        UMUQFAILRETURN("Error type is unknown!");
    }
    return true;
}

/*!
 * \brief set the new error type
 * 
 * \param  ierrorType  Input error type in computing residual
 * 
 * \return true 
 * \return false  if the error type is unknown
 * 
 */
template <typename T>
bool residual<T>::set(int ierrorType)
{
    if (ierrorType == AbsoluteError || ierrorType == ScaledError || ierrorType == SquredError)
    {
        this->errorType = ierrorType;
    }
    else
    {
        UMUQFAILRETURN("Error type is unknown!");
    }
    return true;
}

/*!
 * \brief Compute the residual based on error type
 * 
 * \param observed  Observed data
 * \param predicted Predicted data
 * 
 * \return Residual based on error type
 */
template <typename T>
T residual<T>::operator()(T const &observed, T const &predicted)
{
    switch (this->errorType)
    {
    case AbsoluteError:
        return std::abs(observed - predicted);
    case ScaledError:
        return std::abs(observed - predicted) / std::abs(observed);
    case SquredError:
        return (observed - predicted) * (observed - predicted);
    default:
        return std::abs(observed - predicted);
    }
}

#endif //UMUQ_RESIDUAL_H
