#ifndef UMUQ_ERRORFITNESSTYPE_H
#define UMUQ_ERRORFITNESSTYPE_H

namespace umuq
{

/*! \enum ErrorFitnessTypes
 * \ingroup Numerics_Module
 * 
 * \brief Available Error Fitness type, currently available in %UMUQ
 * 
 */
enum class ErrorFitnessTypes
{
    /*! Sum of the absolute difference between observed and predicted data. */
    errorFitSum,
    /*! Average of the absolute difference between observed and predicted data. */
    errorFitMean,
    /*! Squared root of the average of the absolute difference between observed and predicted data. */
    errorFitRootMean,
    /*! Maximum value of the absolute difference between observed and predicted data. */
    errorFitMax
};

} // namespace umuq

#endif // UMUQ_ERRORFITNESSTYPE
