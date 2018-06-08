#ifndef UMUQ_CURRENT_H
#define UMUQ_CURRENT_H

#include "basic.hpp"

/*!
* \brief current generation structure
*    
* \param queue an integer argument for submission of leaders only
* \param error double argument for measuring error
*/
template<typename T>
struct cgdbp_t : basic<T>
{

    int queue;
    double error;
    /*!
    *  \brief constructor for the default variables
    *  
    * constructor for the default variables it initializes to zero
    */
    cgdbp_t() : queue(0),
                error(0){};
};

/*!
* \brief database generation structure
*    
*/
template<typename T>
struct dbp_t : basic<T>
{
};

/*!
* \brief database generation structure
*    
*/
template<typename T>
struct resdbp_t : basic<T>
{
};

#endif
