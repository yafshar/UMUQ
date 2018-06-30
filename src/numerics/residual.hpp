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
     * \param errorType_ Error type is a residual type (default AbsoluteError) 
     */
    residual(int const errorType_ = AbsoluteError) : errorType(errorType_) {}

    /*!
     * \brief Construct a new residual object
     * 
     * \param errorType_ Error type is a residual type (default AbsoluteError) 
     */
    residual(std::string const &errorType_)
    {
        if (errorType_ == "absolute" || errorType_ == "Absolute" || errorType_ == "AbsoluteError")
        {
            errorType = AbsoluteError;
        }
        else if (errorType_ == "scaled" || errorType_ == "Scaled" || errorType_ == "ScaledError")
        {
            errorType = ScaledError;
        }
        else if (errorType_ == "squared" || errorType_ == "Squared" || errorType_ == "SquredError")
        {
            errorType = SquredError;
        }
        else
        {
			UMUQWARNING("Error type is unknown : Change to default absolute Error!");
            errorType = AbsoluteError;
        }
    }

    /*!
     * \brief set the new error type
     * 
     * \param  errorType_  Error type in computing residual
     * 
     * \return true 
     * \return false  if the error type is unknown 
     */
    bool set(std::string const &errorType_)
    {
        if (errorType_ == "absolute" || errorType_ == "Absolute" || errorType_ == "AbsoluteError")
        {
            errorType = AbsoluteError;
        }
        else if (errorType_ == "scaled" || errorType_ == "Scaled" || errorType_ == "ScaledError")
        {
            errorType = ScaledError;
        }
        else if (errorType_ == "squared" || errorType_ == "Squared" || errorType_ == "SquredError")
        {
            errorType = SquredError;
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
     * \param  errorType_  Error type in computing residual
     * 
     * \return true 
     * \return false  if the error type is unknown 
     */
    bool set(int errorType_)
    {
        if (errorType_ == AbsoluteError || errorType_ == ScaledError || errorType_ == SquredError)
        {
            errorType = errorType_;
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
    T operator()(T const &observed, T const &predicted)
    {
        switch (errorType)
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

  private:
    //! Error type in computing residuals
    int errorType;
};

#endif //UMUQ_RESIDUAL_H
