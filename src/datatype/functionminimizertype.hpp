#ifndef UMUQ_FUNCTIONMINIMIZERTYPE_H
#define UMUQ_FUNCTIONMINIMIZERTYPE_H

namespace umuq
{

inline namespace multimin
{

/*!
 * \ingroup Multimin_Module
 * 
 * \brief Different Function Minimizer, currently available in %UMUQ
 * 
 */
enum class FunctionMinimizerTypes : int
{
    /*! \link umuq::multimin::simplexNM The Simplex method of Nelder and Mead. \endlink  */
    SIMPLEXNM = 1,
    /*! \link umuq::multimin::simplexNM2 The Simplex method of Nelder and Mead (order N operations). \endlink  */
    SIMPLEXNM2 = 2,
    /*! \link umuq::multimin::simplexNM2Rnd The Simplex method of Nelder and Mead (Uses a randomly-oriented set of basis vectors).  \endlink */
    SIMPLEXNM2RND = 3
};

} // namespace multimin
} // namespace umuq

#endif // UMUQ_FUNCTIONMINIMIZERTYPE
