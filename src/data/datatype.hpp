#ifndef UMUQ_DATATYPE_H
#define UMUQ_DATATYPE_H

#include "database.hpp"

/*!
 * \brief Instance of database object for current database
 * 
 */
template<typename T>
database<T> currentData;

/*!
 * \brief Instance of database object for the full database information
 * 
 */
template<typename T>
database<T> fullData;

/*!
 * \brief Instance of database object for experimental data
 * 
 */
template<typename T>
database<T> exprimentData;

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

    pthread_mutex_lock(&currentData<T>.m);
    pos = currentData<T>.idxPos;
    currentData<T>.idxPos++;
    pthread_mutex_unlock(&currentData<T>.m);

    if (pos < currentData<T>.entries)
    {
        std::copy(iParray, iParray + currentData<T>.ndimParray, currentData<T>.Parray.get() + pos * currentData<T>.ndimParray);

        currentData<T>.Fvalue[pos] = *iFvalue;

        if (*ndimGarray > 0)
        {
            std::copy(iGarray, iGarray + currentData<T>.ndimGarray, currentData<T>.Garray.get() + pos * currentData<T>.ndimGarray);
        }

        if (*iSurrogate < std::numeric_limits<int>::max())
        {
            currentData<T>.Surrogate[pos] = *iSurrogate;
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

    pthread_mutex_lock(&fullData<T>.m);
    pos = fullData<T>.idxPos;
    fullData<T>.idxPos++;
    pthread_mutex_unlock(&fullData<T>.m);

    if (pos < fullData<T>.entries)
    {
        std::copy(iParray, iParray + fullData<T>.ndimParray, fullData<T>.Parray.get() + pos * fullData<T>.ndimParray);

        fullData<T>.Fvalue[pos] = *iFvalue;

        if (*ndimGarray > 0)
        {
            std::copy(iGarray, iGarray + fullData<T>.ndimGarray, fullData<T>.Garray.get() + pos * fullData<T>.ndimGarray);
        }

        if (*iSurrogate < std::numeric_limits<int>::max())
        {
            fullData<T>.Surrogate[pos] = *iSurrogate;
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

    pthread_mutex_lock(&exprimentData<T>.m);
    pos = exprimentData<T>.idxPos;
    exprimentData<T>.idxPos++;
    pthread_mutex_unlock(&exprimentData<T>.m);

    if (pos < exprimentData<T>.entries)
    {
        std::copy(iParray, iParray + exprimentData<T>.ndimParray, exprimentData<T>.Parray.get() + pos * exprimentData<T>.ndimParray);

        exprimentData<T>.Fvalue[pos] = *iFvalue;

        if (*ndimGarray > 0)
        {
            std::copy(iGarray, iGarray + exprimentData<T>.ndimGarray, exprimentData<T>.Garray.get() + pos * exprimentData<T>.ndimGarray);
        }

        if (*iSurrogate < std::numeric_limits<int>::max())
        {
            exprimentData<T>.Surrogate[pos] = *iSurrogate;
        }
    }
}

#endif
