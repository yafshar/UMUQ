#ifndef UMUQ_BASIC_H
#define UMUQ_BASIC_H

#include "../misc/array.hpp"

/*!
*  \brief basic structure
*    
*  \param Parray     double array for points in space
*  \param ndimParray an integer argument shows the size of Parray
*  \param Garray     double array
*  \param ndimGarray an integer argument shows the size of Garray
*  \param Fvalue     double argument for the function value
*  \param surrogate  an integer argument shows the surrogate model
*  \param nsel       an integer argument for selection of leaders only
*/
template <typename T>
class basic
{
   /*!
    *  \brief constructor for the default variables
    *    
    */
    basic() : Parray(),
              ndimParray(0),
              Garray(),
              ndimGarray(0),
              Fvalue(0),
              surrogate(0),
              nsel(0){};

  public:
    //! Wrapper to the actual data
    ArrayWrapper<T> Parray;
    int ndimParray;

    //! Wrapper to the actual data
    ArrayWrapper<T> Garray;
    int ndimGarray;

    T Fvalue;

    int surrogate;
    int nsel;

  private:
    std::unique_ptr<T[]> arrayData;
};

#endif
