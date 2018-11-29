#ifndef UMUQ_ERRORTYPE_H
#define UMUQ_ERRORTYPE_H

namespace umuq
{

/*! \enum ErrorTypes
 * \ingroup Numerics_Module
 * 
 * \brief Different residuals Error types, currently available in %UMUQ
 * 
 */
enum class ErrorTypes
{
    /*! Absolute difference between observed and predicted data. */
    AbsoluteError,
    /*! A ratio of absolute difference between observed and predicted data to the absolute value of observed data. */
    ScaledError,
    /*! Squared value of the difference between observed and predicted data. */
    SquaredError
};

} // namespace umuq

#endif // UMUQ_ERRORTYPE
