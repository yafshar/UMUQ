#ifndef UMUQ_GLOBAL_H
#define UMUQ_GLOBAL_H

#include "misc/timer.hpp"

namespace umuq
{
/*! \class global 
 * \brief Encapsulates global objects and functions
 * 
 */

/*!
 * \brief Global timer 
 * 
 * \returns umuqTimer 
 */
#ifdef DEBUG
extern umuqTimer gTimer;
#endif
umuqTimer gTimer(false);

} // namespace umuq

#endif // UMUQ_GLOBAL
