#ifndef UMUQ_GLOBAL_H
#define UMUQ_GLOBAL_H

#include "misc/timer.hpp"
#include "io/pyplot.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"

namespace umuq
{

/*! 
 * \defgroup Global_Module Global module
 * This is the global module of %UMUQ encapsulates global objects necessary in many classes of %UMUQ.
 */

/*!
 * \ingroup Global_Module 
 * 
 * \brief Global timer 
 * 
 * \returns umuqTimer 
 */
#ifdef DEBUG
extern umuq::umuqTimer gTimer;
#endif
umuq::umuqTimer gTimer(false);

/*!
 * \ingroup Global_Module 
 * 
 * \brief Create a global instance of the Pyplot from Pyplot library
 * 
 */
extern umuq::matplotlib_223::pyplot plt;
umuq::matplotlib_223::pyplot plt;

/*!
 * \ingroup Global_Module
 * 
 * \brief Get an instance of a seeded pseudo random object
 */
extern umuq::psrandom prng;
umuq::psrandom prng(12345678);

} // namespace umuq

#endif // UMUQ_GLOBAL
