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
 * \brief computes residuals of observation and predicted data based on Error type
 *
 * List of available Error type:
 *  - \b AbsoluteError Absolute difference between observed data and predicted one
 *  - \b ScaledError   It is a ratio of absolute difference between observed data and predicted one towards 
 *                     absolute value of observation data
 *  - \b SquredError   Squre value of the difference between observed data and predicted one
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
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Error type is unknown : Change to default absolute Error" << std::endl;
            errorType = AbsoluteError;
        }
    }

    /*!
     * \brief set the new error type
     * 
     * \param  errorType_  Error type is a residual type
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
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Error type is unknown : " << std::endl;
            return false;
        }
        return true;
    }

    bool set(int errorType_)
    {
        if (errorType_ == AbsoluteError || errorType_ == ScaledError || errorType_ == SquredError)
        {
            errorType = errorType_;
        }
        else
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Error type is unknown : " << std::endl;
            return false;
        }
        return true;
    }

    /*!
     * \brief compute the residual based on error type
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
        case default:
            return std::abs(observed - predicted);
        }
    }

  private:
    int errorType;
};

#endif //UMUQ_RESIDUAL_H
