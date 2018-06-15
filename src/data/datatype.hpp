#ifndef UMUQ_DATATYPE_H
#define UMUQ_DATATYPE_H

#include "database.hpp"

/*!
 * \brief Instance of database object for current database
 * 
 */
database<double> currentData;

/*!
 * \brief Instance of database object for the full database information
 * 
 */
database<double> fullData;

/*!
 * \brief Instance of database object for experimental data
 * 
 */
database<double> exprimentData;

/*!
 * \brief Updating the data information at each point @iParray 
 * 
 * \tparam T          Data type (T is a floating-point type)
 * 
 * \param iParray     Points or sampling points array
 * \param iFvalue     Function value at the sampling point 
 * \param iGarray     Array of data @iParray 
 * \param ndimGarray  Dimension of G array
 * \param iSurrogate  Surrogate
 */
template <typename T>
void current_updateTask(T const *iParray, T const *iFvalue, T const *iGarray, int const *ndimGarray, int const *iSurrogate)
{
    std::size_t pos;

    pthread_mutex_lock(&currentData.m);
    pos = currentData.idxPos;
    currentData.idxPos++;
    pthread_mutex_unlock(&currentData.m);

    if (pos < currentData.entries)
    {
        std::copy(iParray, iParray + currentData.ndimParray, currentData.Parray.get() + pos * currentData.ndimParray);

        currentData.Fvalue[pos] = *iFvalue;

        if (*ndimGarray > 0)
        {
            std::copy(iGarray, iGarray + currentData.ndimGarray, currentData.Garray.get() + pos * currentData.ndimGarray);
        }

        if (*iSurrogate < std::numeric_limits<int>::max())
        {
            currentData.Surrogate[pos] = *iSurrogate;
        }
    }
}

/*!
 * \brief Updating the data information at each point @iParray 
 * 
 * \tparam T          Data type (T is a floating-point type)
 * 
 * \param iParray     Points or sampling points array
 * \param iFvalue     Function value at the sampling point 
 * \param iGarray     Array of data @iParray 
 * \param ndimGarray  Dimension of G array
 * \param iSurrogate  Surrogate
 */
template <typename T>
void full_updateTask(T const *iParray, T const *iFvalue, T const *iGarray, int const *ndimGarray, int const *iSurrogate)
{
    std::size_t pos;

    pthread_mutex_lock(&fullData.m);
    pos = fullData.idxPos;
    fullData.idxPos++;
    pthread_mutex_unlock(&fullData.m);

    if (pos < fullData.entries)
    {
        std::copy(iParray, iParray + fullData.ndimParray, fullData.Parray.get() + pos * fullData.ndimParray);

        fullData.Fvalue[pos] = *iFvalue;

        if (*ndimGarray > 0)
        {
            std::copy(iGarray, iGarray + fullData.ndimGarray, fullData.Garray.get() + pos * fullData.ndimGarray);
        }

        if (*iSurrogate < std::numeric_limits<int>::max())
        {
            fullData.Surrogate[pos] = *iSurrogate;
        }
    }
}

/*!
 * \brief Updating the data information at each point @iParray 
 * 
 * \tparam T          Data type (T is a floating-point type)
 * 
 * \param iParray     Points or sampling points array
 * \param iFvalue     Function value at the sampling point 
 * \param iGarray     Array of data @iParray 
 * \param ndimGarray  Dimension of G array
 * \param iSurrogate  Surrogate
 */
template <typename T>
void expr_updateTask(T const *iParray, T const *iFvalue, T const *iGarray, int const *ndimGarray, int const *iSurrogate)
{
    std::size_t pos;

    pthread_mutex_lock(&exprimentData.m);
    pos = exprimentData.idxPos;
    exprimentData.idxPos++;
    pthread_mutex_unlock(&exprimentData.m);

    if (pos < exprimentData.entries)
    {
        std::copy(iParray, iParray + exprimentData.ndimParray, exprimentData.Parray.get() + pos * exprimentData.ndimParray);

        exprimentData.Fvalue[pos] = *iFvalue;

        if (*ndimGarray > 0)
        {
            std::copy(iGarray, iGarray + exprimentData.ndimGarray, exprimentData.Garray.get() + pos * exprimentData.ndimGarray);
        }

        if (*iSurrogate < std::numeric_limits<int>::max())
        {
            exprimentData.Surrogate[pos] = *iSurrogate;
        }
    }
}

#endif
