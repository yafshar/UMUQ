#ifndef UMUQ_BASIC_H
#define UMUQ_BASIC_H

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
struct basic
{
    double *Parray;
    int ndimParray;
    double *Garray;
    int ndimGarray;
    double Fvalue;
    int surrogate;
    int nsel;
    /*!
    *  \brief constructor for the default variables
    *    
    */
    basic() : Parray(nullptr),
              ndimParray(0),
              Garray(nullptr),
              ndimGarray(0),
              Fvalue(0),
              surrogate(0),
              nsel(0){};
};

#endif
