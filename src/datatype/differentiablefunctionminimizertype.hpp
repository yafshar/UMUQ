#ifndef UMUQ_DIFFERENTIABLEFUNCTIONMINIMIZERTYPE_H
#define UMUQ_DIFFERENTIABLEFUNCTIONMINIMIZERTYPE_H

namespace umuq
{

inline namespace multimin
{

/*! 
 * \enum differentiableFunctionMinimizerTypes
 * \ingroup Multimin_Module
 * 
 * \brief Different available differentiable Function Minimizer available in %UMUQ
 * 
 */
enum class differentiableFunctionMinimizerTypes : int
{
    /*! \link umuq::multimin::bfgs The Limited memory Broyden-Fletcher-Goldfarb-Shanno method. */
    BFGS = 10,
    /*! \link umuq::multimin::bfgs2 The Limited memory Broyden-Fletcher-Goldfarb-Shanno method (Fletcher's implementation). */
    BFGS2 = 11,
    /*! \link umuq::multimin::conjugateFr The conjugate gradient Fletcher-Reeve algorithm. */
    CONJUGATEFR = 12,
    /*! \link umuq::multimin::conjugatePr The conjugate Polak-Ribiere gradient algorithm. */
    CONJUGATEPR = 13,
    /*! \link umuq::multimin::steepestDescent The steepestDescent for differentiable function minimizer type. */
    STEEPESTDESCENT = 14
};

} // namespace multimin
} // namespace umuq

#endif // UMUQ_DIFFERENTIABLEFUNCTIONMINIMIZERTYPE
