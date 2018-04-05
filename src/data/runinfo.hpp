#ifndef UMHBM_RUNINFO_H
#define UMHBM_RUNINFO_H

/*!
*  \brief run info structure
*    
* \param Gen    
* \param CoefVar        The coefficient of variation of the plausibility weights 
* \param p              cluster-wide
* \param currentuniques   
* \param logselection   
* \param acceptance  
* \param SS             cluster-wide
* \param meantheta  
*/
struct runinfo_t
{
    int Gen;
    double *CoefVar;                     /*[MAXGENS];*/
    double *p;                           /*[MAXGENS];*/
    int *currentuniques;                 /*[MAXGENS];*/
    double *logselection;                /*[MAXGENS];*/
    double *acceptance;                  /*[MAXGENS];*/
    double **SS; /*[PROBDIM][PROBDIM];*/ //
    double **meantheta;                  /*[MAXGENS][PROBDIM];*/
                                         /*!
    *  \brief constructor for the default variables
    *    
    */
    runinfo_t() : Gen(0),
                  CoefVar(nullptr),
                  p(nullptr),
                  currentuniques(nullptr),
                  logselection(nullptr),
                  acceptance(nullptr),
                  SS(nullptr),
                  meantheta(nullptr){};
};

#endif
