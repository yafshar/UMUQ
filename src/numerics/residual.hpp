#ifndef UMUQ_RESIDUAL_H
#define UMUQ_RESIDUAL_H

#include "core/core.hpp"
#include "datatype/errortype.hpp"
#include "misc/parser.hpp"

#include <cmath>

#include <string>

namespace umuq
{

/*! \class residual
 * \ingroup Numerics_Module
 *
 * \brief Computes residuals of observation and predicted data based on different Error type
 *
 * \tparam DataType Data type
 *
 * List of available Error types:<br>
 *  - \b AbsoluteError Absolute difference between observed and predicted data
 *  - \b ScaledError   It is a ratio of absolute difference between observed and predicted data
 *                     to the absolute value of observed data
 *  - \b SquaredError  Squared value of the difference between observed and predicted data
 *
 * \sa umuq::ErrorTypes
 */
template <typename DataType>
class residual
{
  public:
    /*!
     * \brief Construct a new residual object
     *
     * \param ErrorType Input error type is a residual type (default AbsoluteError)
     */
    residual(ErrorTypes const ErrorType = ErrorTypes::AbsoluteError);

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
    bool set(ErrorTypes const ErrorType);

    /*!
     * \brief Compute the residual based on error type
     *
     * \param observed  Observed data
     * \param predicted Predicted data
     *
     * \return Residual based on error type
     */
    inline DataType operator()(DataType const &observed, DataType const &predicted);

  protected:
    /*!
     * \brief Delete a residual object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    residual(residual<DataType> const &) = delete;

    /*!
     * \brief Delete a residual object assignment
     *
     * Avoiding implicit copy assignment.
     */
    residual<DataType> &operator=(residual<DataType> const &) = delete;

  private:
    //! Error type in computing residuals
    ErrorTypes errorType;
};

template <typename DataType>
residual<DataType>::residual(ErrorTypes const ErrorType) : errorType(ErrorType) {}

template <typename DataType>
residual<DataType>::residual(std::string const &ErrorType)
{
    std::string upErrorType(ErrorType);
    {
        umuq::parser p;
        upErrorType = p.toupper(upErrorType);
    }

    if (upErrorType == "ABSOLUTE" || upErrorType == "ABSOLUTEERROR")
    {
        errorType = ErrorTypes::AbsoluteError;
    }
    else if (upErrorType == "SCALED" || upErrorType == "SCALEDERROR")
    {
        errorType = ErrorTypes::ScaledError;
    }
    else if (upErrorType == "SQUARED" || upErrorType == "SQUAREDERROR")
    {
        errorType = ErrorTypes::SquaredError;
    }
    else
    {
        UMUQWARNING("Error type is unknown : Change to the default absolute Error!");

        errorType = ErrorTypes::AbsoluteError;
    }
}

template <typename DataType>
bool residual<DataType>::set(std::string const &ErrorType)
{
    std::string upErrorType(ErrorType);
    {
        umuq::parser p;
        upErrorType = p.toupper(upErrorType);
    }

    if (upErrorType == "ABSOLUTE" || upErrorType == "ABSOLUTEERROR")
    {
        errorType = ErrorTypes::AbsoluteError;
        return true;
    }
    else if (upErrorType == "SCALED" || upErrorType == "SCALEDERROR")
    {
        errorType = ErrorTypes::ScaledError;
        return true;
    }
    else if (upErrorType == "SQUARED" || upErrorType == "SQUAREDERROR")
    {
        errorType = ErrorTypes::SquaredError;
        return true;
    }

    UMUQFAILRETURN("ErrorType is unknown!");
}

template <typename DataType>
bool residual<DataType>::set(ErrorTypes const ErrorType)
{
    if (ErrorType == ErrorTypes::AbsoluteError || ErrorType == ErrorTypes::ScaledError || ErrorType == ErrorTypes::SquaredError)
    {
        errorType = ErrorType;
        return true;
    }
    else
    {
        UMUQFAILRETURN("ErrorType is unknown!");
    }
}

template <typename DataType>
inline DataType residual<DataType>::operator()(DataType const &observed, DataType const &predicted)
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
