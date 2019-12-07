#ifndef UMUQ_GLOBAL_H
#define UMUQ_GLOBAL_H

#include "misc/timer.hpp"

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

} // namespace umuq

#endif // UMUQ_GLOBAL
