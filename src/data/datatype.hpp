#ifndef UMUQ_DATATYPE_H
#define UMUQ_DATATYPE_H

#include "database.hpp"

/*!
 * \brief Instance of database object 
 * 
 */
template<typename T>
database<T> Data1;

/*!
 * \brief Instance of database object 
 * 
 */
template<typename T>
database<T> Data2;

/*!
 * \brief Instance of database object
 * 
 */
template<typename T>
database<T> Data3;

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
void updateTask1(T const *iParray, T const *iFvalue, T const *iGarray, int const *ndimGarray, int const *iSurrogate)
{
    std::size_t pos;

    pthread_mutex_lock(&Data1<T>.m);
    pos = Data1<T>.idxPos;
    Data1<T>.idxPos++;
    pthread_mutex_unlock(&Data1<T>.m);

    if (pos < Data1<T>.entries)
    {
        std::copy(iParray, iParray + Data1<T>.ndimParray, Data1<T>.Parray.get() + pos * Data1<T>.ndimParray);

        Data1<T>.Fvalue[pos] = *iFvalue;

        if (*ndimGarray > 0)
        {
            std::copy(iGarray, iGarray + Data1<T>.ndimGarray, Data1<T>.Garray.get() + pos * Data1<T>.ndimGarray);
        }

        if (*iSurrogate < std::numeric_limits<int>::max())
        {
            Data1<T>.Surrogate[pos] = *iSurrogate;
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
void updateTask2(T const *iParray, T const *iFvalue, T const *iGarray, int const *ndimGarray, int const *iSurrogate)
{
    std::size_t pos;

    pthread_mutex_lock(&Data2<T>.m);
    pos = Data2<T>.idxPos;
    Data2<T>.idxPos++;
    pthread_mutex_unlock(&Data2<T>.m);

    if (pos < Data2<T>.entries)
    {
        std::copy(iParray, iParray + Data2<T>.ndimParray, Data2<T>.Parray.get() + pos * Data2<T>.ndimParray);

        Data2<T>.Fvalue[pos] = *iFvalue;

        if (*ndimGarray > 0)
        {
            std::copy(iGarray, iGarray + Data2<T>.ndimGarray, Data2<T>.Garray.get() + pos * Data2<T>.ndimGarray);
        }

        if (*iSurrogate < std::numeric_limits<int>::max())
        {
            Data2<T>.Surrogate[pos] = *iSurrogate;
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
void updateTask3(T const *iParray, T const *iFvalue, T const *iGarray, int const *ndimGarray, int const *iSurrogate)
{
    std::size_t pos;

    pthread_mutex_lock(&Data3<T>.m);
    pos = Data3<T>.idxPos;
    Data3<T>.idxPos++;
    pthread_mutex_unlock(&Data3<T>.m);

    if (pos < Data3<T>.entries)
    {
        std::copy(iParray, iParray + Data3<T>.ndimParray, Data3<T>.Parray.get() + pos * Data3<T>.ndimParray);

        Data3<T>.Fvalue[pos] = *iFvalue;

        if (*ndimGarray > 0)
        {
            std::copy(iGarray, iGarray + Data3<T>.ndimGarray, Data3<T>.Garray.get() + pos * Data3<T>.ndimGarray);
        }

        if (*iSurrogate < std::numeric_limits<int>::max())
        {
            Data3<T>.Surrogate[pos] = *iSurrogate;
        }
    }
}

#endif
