#ifndef UMUQ_STATS_H
#define UMUQ_STATS_H

#include "../misc/arraywrapper.hpp"

namespace umuq
{

/*! \class stats
 * \ingroup Numerics_Module
 * 
 * \brief stats is a class which includes some functionality for statistics of the input data
 *
 * It includes:
 * - \b minelement         Finds the smallest element in the array of data
 * - \b maxelement         Finds the greatest element in the array of data
 * - \b minelement_index   Finds the position of the smallest element in the array of data 
 * - \b maxelement_index   Finds the position of the greatest element in the array of data 
 * - \b sum                Computes the sum of the elements in the array of data
 * - \b sumAbs             Computes the sum of the absolute value of the elements in the array of data
 * - \b mean               Computes the mean of the elements in the array of data
 * - \b median             Computes the median of the elements in the array of data
 * - \b medianAbs          Computes the median absolute deviation (MAD) of the elements in the array of data
 * - \b stddev             Computes the standard deviation of the elements in the array of data
 * - \b coefvar            Computes the coefficient of variation (CV)
 * - \b minmaxNormal       Scales the numeric data using the MinMax normalization method
 * - \b zscoreNormal       Scales the numeric data using the Z-score normalization method
 * - \b robustzscoreNormal Scales the numeric data using the robust Z-score normalization method
 * - \b covariance         Compute the covariance
 * - \b unique             Eliminates all but the first element from every consecutive sample points,
 *                         Find the unique n-dimensions sample points in an array of nRows * nCols data
 */
struct stats
{
    /*!
     * \brief Construct a new stats object
     * 
     */
    stats();

    /*!
     * \brief Destroy the stats object
     * 
     */
    ~stats();

    /*!
     * \brief Finds the smallest element in the array of data with stride
     * 
     * \tparam T data type
     * 
     * \param idata  array of data
     * \param nSize  size of the array
     * \param Stride element stride (default is 1)
     * 
     * \returns The smallest element in the array of data
     */
    template <typename T>
    inline T minelement(T const *idata, int const nSize, int const Stride = 1) const;

    template <typename T>
    inline T minelement(std::vector<T> const &idata) const;

    template <typename T>
    inline T minelement(arrayWrapper<T> const &iArray) const;

    /*!
     * \brief Finds the greatest element in the array of data with stride
     * 
     * \tparam T data type
     * 
     * \param idata array of data
     * \param nSize size of the array
     * \param Stride element stride (default is 1)
     * 
     * \returns The greatest element in the array of data
     */
    template <typename T>
    inline T maxelement(T const *idata, int const nSize, int const Stride = 1) const;

    template <typename T>
    inline T maxelement(std::vector<T> const &idata) const;

    template <typename T>
    inline T maxelement(arrayWrapper<T> const &iArray) const;

    /*!
     * \brief Finds the position of the smallest element in the array of data (idata) with stride
     * 
     * \tparam T data type
     * 
     * \param idata array of data
     * \param nSize size of the array
     * \param Stride element stride (default is 1)
     * 
     * \returns The the position of the smallest element
     */
    template <typename T>
    inline int minelement_index(T const *idata, int const nSize, int const Stride = 1) const;

    template <typename T>
    inline int minelement_index(std::vector<T> const &idata) const;

    template <typename T>
    inline int minelement_index(arrayWrapper<T> const &iArray) const;

    /*!
     * \brief Finds the position of the greatest element in the array of data (idata) with Stride
     * 
     * \tparam T data type
     * 
     * \param idata array of data
     * \param nSize size of the array
     * \param Stride element stride (default is 1)
     * 
     * \returns The the position of the greatest element
     */
    template <typename T>
    inline int maxelement_index(T const *idata, int const nSize, int const Stride = 1) const;

    template <typename T>
    inline int maxelement_index(std::vector<T> const &idata) const;

    template <typename T>
    inline int maxelement_index(arrayWrapper<T> const &iArray) const;

    /*!
     * \brief Computes the sum of the elements in the array of data with stride
     * 
     * \tparam T    Data type
     * \tparam TOut Data type of the return output result (default is double)
     * 
     * \param idata  Array of data
     * \param nSize  Size of the array
     * \param Stride element stride (default is 1)
     * 
     * \returns The sum of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut sum(T const *idata, int const nSize, int const Stride = 1) const;

    template <typename T, typename TOut = double>
    inline TOut sum(std::vector<T> const &idata) const;

    template <typename T, typename TOut = double>
    inline TOut sum(arrayWrapper<T> const &iArray) const;

    /*!
     * \brief Computes the sum of the absolute value of the elements in the array of data with stride
     * 
     * \tparam T    Data type
     * \tparam TOut Data type of the return output result (default is double)
     * 
     * \param idata  Array of data
     * \param nSize  Size of the array
     * \param Stride Element stride (default is 1)
     * 
     * \returns The sum of the absolute value of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut sumAbs(T const *idata, int const nSize, int const Stride = 1) const;

    template <typename T, typename TOut = double>
    inline TOut sumAbs(std::vector<T> const &idata) const;

    template <typename T, typename TOut = double>
    inline TOut sumAbs(arrayWrapper<T> const &iArray) const;

    /*!
     * \brief Computes the mean of the elements in the array of data with stride
     * 
     * \tparam T data type
     * \tparam TOut type of return output result (default is double)
     * 
     * \param idata array of data
     * \param nSize size of the array
     * \param Stride element stride (default is 1)
     * 
     * \returns The mean of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut mean(T const *idata, int const nSize, int const Stride = 1) const;

    template <typename T, typename TOut = double>
    inline TOut mean(std::vector<T> const &idata) const;

    template <typename T, typename TOut = double>
    inline TOut mean(arrayWrapper<T> const &iArray) const;

    /*!
     * \brief Computes the median of the elements in the array of data with Stride
     * 
     * \tparam T data type
     * \tparam TOut type of return output result (default is double)
     * 
     * \param idata array of data
     * \param nSize size of the array
     * \param Stride element stride (default is 1)
     * 
     * \returns The median of the elements in the array of data with Stride
     */
    template <typename T, typename TOut = double>
    inline TOut median(T const *idata, int const nSize, int const Stride = 1);

    template <typename T, typename TOut = double>
    inline TOut median(std::vector<T> const &idata);

    template <typename T, typename TOut = double>
    inline TOut median(arrayWrapper<T> const &iArray);

    /*!
     * \brief Computes the median absolute deviation (MAD) of the elements in the array of data
     * 
     * \tparam T data type
     * \tparam TOut type of return output result (default is double)
     * 
     * \param idata   array of data
     * \param nSize   size of the array
     * \param Stride  element stride
     * \param median_ median of the elements in the array of data
     * 
     * \returns The median absolute deviation of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut medianAbs(T const *idata, int const nSize, int const Stride = 1, TOut &median_ = TOut{});

    template <typename T, typename TOut = double>
    inline TOut medianAbs(std::vector<T> const &idata, TOut &median_ = TOut{});

    template <typename T, typename TOut = double>
    inline TOut medianAbs(arrayWrapper<T> const &iArray, TOut &median_ = TOut{});

    /*!
     * \brief Computes the standard deviation of the elements in the array of data with or without stride
     * 
     * \tparam T    data type
     * \tparam TOut data type of return output result (default is double)
     * 
     * \param idata     array of data
     * \param nSize     size of the array
     * \param Stride    element stride (optional, default is 1)
     * \param idataMean mean of the elements in idata (optional)
     * 
     * \returns The standard deviation of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut stddev(T const *idata, int const nSize, int const Stride = 1, TOut const idataMean = std::numeric_limits<TOut>::max()) const;

    template <typename T, typename TOut = double>
    inline TOut stddev(std::vector<T> const &idata, TOut const idataMean = std::numeric_limits<TOut>::max()) const;

    template <typename T, typename TOut = double>
    inline TOut stddev(arrayWrapper<T> const &iArray, TOut const idataMean = std::numeric_limits<TOut>::max()) const;

    /*!
     * \brief Computes the coefficient of variation (CV), or relative standard deviation (RSD).
     * It is a standardized measure of dispersion of a probability distribution or frequency distribution.
     * It is defined as the ratio of the standard deviation \f$ \sigma \f$ to the mean \f$ \mu \f$ 
     * 
     * \tparam T    data type
     * \tparam TOut data type of return output result (default is double)
     * 
     * \param idata     array of data
     * \param nSize     size of the array
     * \param Stride    element stride (optional, default is 1)
     * \param idataMean mean of the elements in idata (optional)
     * 
     * \returns The coefficient of variation (CV)
     */
    template <typename T, typename TOut = double>
    inline TOut coefvar(T const *idata, int const nSize, int const Stride = 1, TOut const idataMean = std::numeric_limits<TOut>::max()) const;

    template <typename T, typename TOut = double>
    inline TOut coefvar(std::vector<T> const &idata, TOut const idataMean = std::numeric_limits<TOut>::max()) const;

    template <typename T, typename TOut = double>
    inline TOut coefvar(arrayWrapper<T> const &iArray, TOut const idataMean = std::numeric_limits<TOut>::max()) const;

    /*!
     * \brief minmaxNormal scales the numeric data using the MinMax normalization method
     * 
     * Using the MinMax normalization method, one can normalize the values to be between 0 and 1. 
     * Doing so allows to compare values on very different scales to one another by reducing 
     * the dominance of one dimension over the other.
     * 
     * \tparam T data type
     * 
     * \param idata  array of data
     * \param nSize  size of array
     * \param Stride element stride (default is 1)
     */
    template <typename T>
    void minmaxNormal(T *idata, int const nSize, int const Stride = 1);

    template <typename T>
    void minmaxNormal(std::vector<T> &idata);

    /*!
     * \brief zscoreNormal scales the numeric data using the Z-score normalization method
     * 
     * Using the Z-score normalization method, one can normalize the values to be the number of 
     * standard deviations an observation is from the mean of each dimension. 
     * This allows to compare data to a normally distributed random variable.
     * 
     * \tparam T Data type
     * 
     * \param idata  Input data
     * \param nSize  Size of array
     * \param Stride Element stride (default is 1)
     */
    template <typename T>
    inline void zscoreNormal(T *idata, int const nSize, int const Stride = 1);

    template <typename T>
    inline void zscoreNormal(std::vector<T> &idata);

    /*!
     * \brief robustzscoreNormal scales the numeric data using the robust Z-score normalization method
     * 
     * Using the robust Z-score normalization method, one can lessen the influence of outliers 
     * on Z-score calculations. Robust Z-score normalization uses the median value as opposed 
     * to the mean value used in Z-score. <br>
     * By using the median instead of the mean, it helps remove some of the influence of outliers 
     * in the data.
     * 
     * 
     * \tparam T Data type
     * 
     * \param idata  Input data
     * \param nSize  Size of the array
     * \param Stride Element stride (default is 1)
     */
    template <typename T>
    inline void robustzscoreNormal(T *idata, int const nSize, int const Stride = 1);

    template <typename T>
    inline void robustzscoreNormal(std::vector<T> &idata);

    /*!
     * \brief Compute the covariance between idata and jdata vectors which must both be of the same length nSize
     * \f$ covariance(idata, jdata) = \frac{1}{n-1} \sum_{i=1}^n (idata_i-iMean)(jdata_i-jMean) \f$
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param idata  Array of data 
     * \param jdata  Array of data
     * \param nSize  Size of array
     * \param iMean  Mean of idata array
     * \param jMean  Mean of jdata array
     * 
     * \returns Covariance (scaler value) between idata and jdata vectors     
     */
    template <typename T, typename TOut = double>
    TOut covariance(T const *idata, T const *jdata, int const nSize, T const iMean, T const jMean);

    /*!
     * \brief Compute the covariance between two arrays of data which must both be of the same length
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param iArray  Array of data 
     * \param jArray  Array of data
     * \param iMean   Mean of iArray 
     * \param jMean   Mean of jArray
     * 
     * \returns Covariance (scaler value) between idata and jdata vectors     
     */
    template <typename T, typename TOut = double>
    TOut covariance(arrayWrapper<T> const &iArray, arrayWrapper<T> const &jArray, T const iMean, T const jMean);

    template <typename T, typename TOut = double>
    TOut covariance(std::vector<T> const &idata, std::vector<T> const &jdata, T const iMean, T const jMean);

    /*!
     * \brief Compute the covariance between idata and jdata vectors which must both be of the same length nSize
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param idata  Array of data 
     * \param jdata  Array of data
     * \param nSize  Size of array
     * \param Stride Stride of the data in the array (default is 1)
     * 
     * \returns Covariance (scaler value) between idata and jdata vectors     
     */
    template <typename T, typename TOut = double>
    TOut covariance(T const *idata, T const *jdata, int const nSize, int const Stride = 1);

    /*!
     * \brief Compute the covariance between two arrays of data which must both be of the same length
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param iArray  Array of data 
     * \param jArray  Array of data
     * 
     * \returns Covariance (scaler value) between idata and jdata vectors     
     */
    template <typename T, typename TOut = double>
    TOut covariance(arrayWrapper<T> const &iArray, arrayWrapper<T> const &jArray);

    template <typename T, typename TOut = double>
    TOut covariance(std::vector<T> const &iArray, std::vector<T> const &jArray);

    /*!
     * \brief Compute the covariance array of N-dimensional idata
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param idata  Array of N-dimensional data 
     * \param nSize  Total size of the array
     * \param nDim   Data dimension
     * \param Stride Stride of the data in the array (default is 1). 
     * 
     * The reason for having parameter stride is the case where we have coordinates and function value 
     * and would like to avoid unnecessary copying the data 
     * 
     * \returns Covariance (array of N by N) from N-dimensional idata
     */
    template <typename T, typename TOut = double>
    TOut *covariance(T const *idata, int const nSize, int const nDim, int const Stride = 1);

    /*!
     * \brief Compute the covariance array of N-dimensional idata
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param idata  Array of N-dimensional data with size of [nSize/nDim][nDim]
     * \param nSize  Total size of the array
     * \param nDim   Data dimension
     * \param iMean  Mean of each column or dimension of the array idata with size of [nDim]
     * 
     * \returns Covariance (array of [nDim * nDim]) from N-dimensional idata
     */
    template <typename T, typename TOut = double>
    TOut *covariance(T const *idata, int const nSize, int const nDim, T const *iMean);

    /*!
     * \brief Eliminates all but the first element from every consecutive sample points of dimension n = nCols
     * of equivalent elements from idata which is an array of size nRows * nCols.
     * Find the unique n-dimensions sample points in an array of nRows * nCols data.
     * 
     * \tparam T     Data type
     * 
     * \param idata  Input data
     * \param nRows  Number of rows
     * \param nCols  Number of columns (data dimension)
     * \param udata  Unique data (every row in this data is unique)
     */
    template <typename T>
    void unique(T const *idata, int const nRows, int const nCols, std::vector<T> &udata);

    template <typename T>
    void unique(std::vector<T> const &idata, int const nRows, int const nCols, std::vector<T> &udata);

    template <typename T>
    void unique(std::unique_ptr<T[]> const &idata, int const nRows, int const nCols, std::vector<T> &udata);
};

stats::stats()
{
    if (!(unrolledIncrement == 4 || unrolledIncrement == 6 || unrolledIncrement == 8 || unrolledIncrement == 10 || unrolledIncrement == 12))
    {
        UMUQFAIL("The unrolled Increment value is not correctly set!");
    }
}

stats::~stats() {}

template <typename T>
inline T stats::minelement(T const *idata, int const nSize, int const Stride) const
{
    if (Stride > 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? *std::min_element(iArray.begin(), iArray.end()) : throw(std::runtime_error("Wrong input size!"));
    }
    return (nSize > 0) ? *std::min_element(idata, idata + nSize) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline T stats::minelement(std::vector<T> const &idata) const
{
    return (idata.size() > 0) ? *std::min_element(idata.begin(), idata.end()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline T stats::minelement(arrayWrapper<T> const &iArray) const
{
    return (iArray.size() > 0) ? *std::min_element(iArray.begin(), iArray.end()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline T stats::maxelement(T const *idata, int const nSize, int const Stride) const
{
    if (Stride > 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? *std::max_element(iArray.begin(), iArray.end()) : throw(std::runtime_error("Wrong input size!"));
    }
    return (nSize > 0) ? *std::max_element(idata, idata + nSize) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline T stats::maxelement(std::vector<T> const &idata) const
{
    return (idata.size() > 0) ? *std::max_element(idata.begin(), idata.end()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline T stats::maxelement(arrayWrapper<T> const &iArray) const
{
    return (iArray.size() > 0) ? *std::max_element(iArray.begin(), iArray.end()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline int stats::minelement_index(T const *idata, int const nSize, int const Stride) const
{
    if (Stride > 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? static_cast<int>(std::distance(iArray.begin(), std::min_element(iArray.begin(), iArray.end())) * Stride) : throw(std::runtime_error("Wrong input size!"));
    }

    return (nSize > 0) ? static_cast<int>(std::distance(idata, std::min_element(idata, idata + nSize))) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline int stats::minelement_index(std::vector<T> const &idata) const
{
    return (idata.size() > 0) ? static_cast<int>(std::distance(idata.begin(), std::min_element(idata.begin(), idata.end()))) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline int stats::minelement_index(arrayWrapper<T> const &iArray) const
{
    return (iArray.size() > 0) ? static_cast<int>(std::distance(iArray.begin(), std::min_element(iArray.begin(), iArray.end())) * iArray.stride()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline int stats::maxelement_index(T const *idata, int const nSize, int const Stride) const
{
    if (Stride > 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? static_cast<int>(std::distance(iArray.begin(), std::max_element(iArray.begin(), iArray.end())) * Stride) : throw(std::runtime_error("Wrong input size!"));
    }
    return (nSize > 0) ? static_cast<int>(std::distance(idata, std::max_element(idata, idata + nSize))) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline int stats::maxelement_index(std::vector<T> const &idata) const
{
    return (idata.size() > 0) ? static_cast<int>(std::distance(idata.begin(), std::max_element(idata.begin(), idata.end()))) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T>
inline int stats::maxelement_index(arrayWrapper<T> const &iArray) const
{
    return (iArray.size() > 0) ? static_cast<int>(std::distance(iArray.begin(), std::max_element(iArray.begin(), iArray.end())) * iArray.stride()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T, typename TOut>
inline TOut stats::sum(T const *idata, int const nSize, int const Stride) const
{
    if (nSize <= 0)
    {
        return TOut{};
    }
    if (Stride > 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{})) : TOut{};
    }
#if unrolledIncrement == 0
    return static_cast<TOut>(std::accumulate(idata, idata + nSize, T{}));
#else
    T s(0);
    int const m = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (m)
    {
        for (int i = 0; i < m; i++)
        {
            s += idata[i];
        }
    }
    for (int i = m; i < nSize; i += unrolledIncrement)
    {
#if unrolledIncrement == 4
        s += idata[i] + idata[i + 1] + idata[i + 2] + idata[i + 3];
#endif
#if unrolledIncrement == 6
        s += idata[i] + idata[i + 1] + idata[i + 2] + idata[i + 3] + idata[i + 4] + idata[i + 5];
#endif
#if unrolledIncrement == 8
        s += idata[i] + idata[i + 1] + idata[i + 2] + idata[i + 3] + idata[i + 4] + idata[i + 5] + idata[i + 6] + idata[i + 7];
#endif
#if unrolledIncrement == 10
        s += idata[i] + idata[i + 1] + idata[i + 2] + idata[i + 3] + idata[i + 4] + idata[i + 5] + idata[i + 6] + idata[i + 7] + idata[i + 8] + idata[i + 9];
#endif
#if unrolledIncrement == 12
        s += idata[i] + idata[i + 1] + idata[i + 2] + idata[i + 3] + idata[i + 4] + idata[i + 5] + idata[i + 6] + idata[i + 7] + idata[i + 8] + idata[i + 9] + idata[i + 10] + idata[i + 11];
#endif
    }
    return static_cast<TOut>(s);
#endif
}

template <typename T, typename TOut>
inline TOut stats::sum(std::vector<T> const &idata) const
{
    return (idata.size() > 0) ? (static_cast<TOut>(std::accumulate(idata.begin(), idata.end(), T{}))) : TOut{};
}

template <typename T, typename TOut>
inline TOut stats::sum(arrayWrapper<T> const &iArray) const
{
    return (iArray.size() > 0) ? (static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{}))) : TOut{};
}

template <typename T, typename TOut>
inline TOut stats::sumAbs(T const *idata, int const nSize, int const Stride) const
{
    if (nSize <= 0)
    {
        return TOut{};
    }
    if (Stride > 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{}, [](T const &lhs, T const &rhs) { return lhs + std::abs(rhs); })) : TOut{};
    }
#if unrolledIncrement == 0
    return static_cast<TOut>(std::accumulate(idata, idata + nSize, T{}, [](T const &lhs, T const &rhs) { return lhs + std::abs(rhs); }));
#else
    T s(0);
    int const m = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (m)
    {
        for (int i = 0; i < m; i++)
        {
            s += std::abs(idata[i]);
        }
    }
    for (int i = m; i < nSize; i += unrolledIncrement)
    {
#if unrolledIncrement == 4
        s += std::abs(idata[i]) +
             std::abs(idata[i + 1]) +
             std::abs(idata[i + 2]) +
             std::abs(idata[i + 3]);
#endif
#if unrolledIncrement == 6
        s += std::abs(idata[i]) +
             std::abs(idata[i + 1]) +
             std::abs(idata[i + 2]) +
             std::abs(idata[i + 3]) +
             std::abs(idata[i + 4]) +
             std::abs(idata[i + 5]);
#endif
#if unrolledIncrement == 8
        s += std::abs(idata[i]) +
             std::abs(idata[i + 1]) +
             std::abs(idata[i + 2]) +
             std::abs(idata[i + 3]) +
             std::abs(idata[i + 4]) +
             std::abs(idata[i + 5]) +
             std::abs(idata[i + 6]) +
             std::abs(idata[i + 7]);
#endif
#if unrolledIncrement == 10
        s += std::abs(idata[i]) +
             std::abs(idata[i + 1]) +
             std::abs(idata[i + 2]) +
             std::abs(idata[i + 3]) +
             std::abs(idata[i + 4]) +
             std::abs(idata[i + 5]) +
             std::abs(idata[i + 6]) +
             std::abs(idata[i + 7]) +
             std::abs(idata[i + 8]) +
             std::abs(idata[i + 9]);
#endif
#if unrolledIncrement == 12
        s += std::abs(idata[i]) +
             std::abs(idata[i + 1]) +
             std::abs(idata[i + 2]) +
             std::abs(idata[i + 3]) +
             std::abs(idata[i + 4]) +
             std::abs(idata[i + 5]) +
             std::abs(idata[i + 6]) +
             std::abs(idata[i + 7]) +
             std::abs(idata[i + 8]) +
             std::abs(idata[i + 9]) +
             std::abs(idata[i + 10]) +
             std::abs(idata[i + 11]);
#endif
    }
    return static_cast<TOut>(s);
#endif
}

template <typename T, typename TOut>
inline TOut stats::sumAbs(std::vector<T> const &idata) const
{
    return (idata.size() > 0) ? (static_cast<TOut>(std::accumulate(idata.begin(), idata.end(), T{}, [](T const &lhs, T const &rhs) { return lhs + std::abs(rhs); }))) : TOut{};
}

template <typename T, typename TOut>
inline TOut stats::sumAbs(arrayWrapper<T> const &iArray) const
{
    return (iArray.size() > 0) ? (static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{}, [](T const &lhs, T const &rhs) { return lhs + std::abs(rhs); }))) : TOut{};
}

template <typename T, typename TOut>
inline TOut stats::mean(T const *idata, int const nSize, int const Stride) const
{
    if (nSize <= 0)
    {
        UMUQFAIL("Wrong input size!");
    }
    if (Stride > 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? (static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{})) / iArray.size()) : throw(std::runtime_error("Wrong input size!"));
    }
    return sum<T, TOut>(idata, nSize) / static_cast<TOut>(nSize);
}

template <typename T, typename TOut>
inline TOut stats::mean(std::vector<T> const &idata) const
{
    return (idata.size() > 0) ? (static_cast<TOut>(std::accumulate(idata.begin(), idata.end(), T{})) / idata.size()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T, typename TOut>
inline TOut stats::mean(arrayWrapper<T> const &iArray) const
{
    return (iArray.size() > 0) ? (static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{})) / iArray.size()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename T, typename TOut>
inline TOut stats::median(T const *idata, int const nSize, int const Stride)
{
    if (Stride > 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);

        // We do partial sorting algorithm that rearranges elements
        std::vector<TOut> data(iArray.begin(), iArray.end());
        std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
        return (iArray.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
    }

    // We do partial sorting algorithm that rearranges elements
    std::vector<TOut> data(idata, idata + nSize);
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (nSize > 1) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename T, typename TOut>
inline TOut stats::median(std::vector<T> const &idata)
{
    // We do partial sorting algorithm that rearranges elements
    std::vector<TOut> data(idata);

    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (idata.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename T, typename TOut>
inline TOut stats::median(arrayWrapper<T> const &iArray)
{
    // We do partial sorting algorithm that rearranges elements
    std::vector<TOut> data(iArray.begin(), iArray.end());

    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (iArray.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename T, typename TOut>
inline TOut stats::medianAbs(T const *idata, int const nSize, int const Stride, TOut &median_)
{
    arrayWrapper<T> iArray(idata, nSize, Stride);

    // We do partial sorting algorithm that rearranges elements
    std::vector<TOut> data(iArray.begin(), iArray.end());

    // std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    median_ = data[data.size() / 2];
    std::for_each(data.begin(), data.end(), [&](TOut &d_i) { d_i = std::abs(d_i - median_); });

    // std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (nSize > 1) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename T, typename TOut>
inline TOut stats::medianAbs(std::vector<T> const &idata, TOut &median_)
{
    // We do partial sorting algorithm that rearranges elements
    std::vector<TOut> data(idata);

    //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    median_ = data[data.size() / 2];
    std::for_each(data.begin(), data.end(), [&](TOut &d_i) { d_i = std::abs(d_i - median_); });

    //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (idata.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename T, typename TOut>
inline TOut stats::medianAbs(arrayWrapper<T> const &iArray, TOut &median_)
{
    // We do partial sorting algorithm that rearranges elements
    std::vector<TOut> data(iArray.begin(), iArray.end());

    //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    median_ = data[data.size() / 2];
    std::for_each(data.begin(), data.end(), [&](TOut &d_i) { d_i = std::abs(d_i - median_); });

    //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (iArray.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename T, typename TOut>
inline TOut stats::stddev(T const *idata, int const nSize, int const Stride, TOut const idataMean) const
{
    TOut const m = idataMean < std::numeric_limits<TOut>::max() ? idataMean : mean<T, TOut>(idata, nSize, Stride);
    TOut s(0);
    if (Stride != 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);
        std::for_each(iArray.begin(), iArray.end(), [&](T const d) { s += (d - m) * (d - m); });
        return (iArray.size() > 1) ? std::sqrt(s / (iArray.size() - 1)) : std::sqrt(s);
    }
#if unrolledIncrement == 0
    std::for_each(idata, idata + nSize, [&](T const d) { s += (d - m) * (d - m); });
#else
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            s += (idata[i] - m) * (idata[i] - m);
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
    {
        TOut const diff0 = idata[i] - m;
        TOut const diff1 = idata[i + 1] - m;
        TOut const diff2 = idata[i + 2] - m;
        TOut const diff3 = idata[i + 3] - m;
#if unrolledIncrement == 4
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
#endif
#if unrolledIncrement == 6
        TOut const diff4 = idata[i + 4] - m;
        TOut const diff5 = idata[i + 5] - m;
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5;
#endif
#if unrolledIncrement == 8
        TOut const diff4 = idata[i + 4] - m;
        TOut const diff5 = idata[i + 5] - m;
        TOut const diff6 = idata[i + 6] - m;
        TOut const diff7 = idata[i + 7] - m;
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
#endif
#if unrolledIncrement == 10
        TOut const diff4 = idata[i + 4] - m;
        TOut const diff5 = idata[i + 5] - m;
        TOut const diff6 = idata[i + 6] - m;
        TOut const diff7 = idata[i + 7] - m;
        TOut const diff8 = idata[i + 8] - m;
        TOut const diff9 = idata[i + 9] - m;
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9;
#endif
#if unrolledIncrement == 12
        TOut const diff4 = idata[i + 4] - m;
        TOut const diff5 = idata[i + 5] - m;
        TOut const diff6 = idata[i + 6] - m;
        TOut const diff7 = idata[i + 7] - m;
        TOut const diff8 = idata[i + 8] - m;
        TOut const diff9 = idata[i + 9] - m;
        TOut const diff10 = idata[i + 10] - m;
        TOut const diff11 = idata[i + 11] - m;
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9 + diff10 * diff10 + diff11 * diff11;
#endif
    }
#endif
    return (nSize > 1) ? std::sqrt(s / (nSize - 1)) : std::sqrt(s);
}

template <typename T, typename TOut>
inline TOut stats::stddev(std::vector<T> const &idata, TOut const idataMean) const
{
    TOut const m = idataMean < std::numeric_limits<TOut>::max() ? idataMean : mean<T, TOut>(idata);
    TOut s(0);
    std::for_each(idata.begin(), idata.end(), [&](T const d) { s += (d - m) * (d - m); });
    return (idata.size() > 1) ? std::sqrt(s / (idata.size() - 1)) : std::sqrt(s);
}

template <typename T, typename TOut>
inline TOut stats::stddev(arrayWrapper<T> const &iArray, TOut const idataMean) const
{
    TOut const m = idataMean < std::numeric_limits<TOut>::max() ? idataMean : mean<T, TOut>(iArray);
    TOut s(0);
    std::for_each(iArray.begin(), iArray.end(), [&](T const d) { s += (d - m) * (d - m); });
    return (iArray.size() > 1) ? std::sqrt(s / (iArray.size() - 1)) : std::sqrt(s);
}

template <typename T, typename TOut>
inline TOut stats::coefvar(T const *idata, int const nSize, int const Stride, TOut const idataMean) const
{
    TOut const m = idataMean < std::numeric_limits<TOut>::max() ? idataMean : mean<T, TOut>(idata, nSize, Stride);
    TOut const s = stddev<T, TOut>(idata, nSize, Stride, m);
    return s / m;
}

template <typename T, typename TOut>
inline TOut stats::coefvar(std::vector<T> const &idata, TOut const idataMean) const
{
    TOut const m = idataMean < std::numeric_limits<TOut>::max() ? idataMean : mean<T, TOut>(idata);
    TOut s(0);
    std::for_each(idata.begin(), idata.end(), [&](T const d) { s += (d - m) * (d - m); });
    return (idata.size() > 1) ? std::sqrt(s / (idata.size() - 1)) / m : std::sqrt(s) / m;
}

template <typename T, typename TOut>
inline TOut stats::coefvar(arrayWrapper<T> const &iArray, TOut const idataMean) const
{
    TOut const m = idataMean < std::numeric_limits<TOut>::max() ? idataMean : mean<T, TOut>(iArray);
    TOut s(0);
    std::for_each(iArray.begin(), iArray.end(), [&](T const d) { s += (d - m) * (d - m); });
    return (iArray.size() > 1) ? std::sqrt(s / (iArray.size() - 1)) / m : std::sqrt(s) / m;
}

template <typename T>
void stats::minmaxNormal(T *idata, int const nSize, int const Stride)
{
    T const MinValue = minelement<T>(idata, nSize, Stride);
    T const MaxValue = maxelement<T>(idata, nSize, Stride);

    T denom = MaxValue - MinValue;
    if (denom < std::numeric_limits<T>::epsilon())
    {
        UMUQWARNING("Maximum and Minimum Value are identical!");
        denom = std::numeric_limits<T>::epsilon();
    }
    if (Stride > 1)
    {
        for (int i = 0; i < nSize; i += Stride)
        {
            idata[i] -= MinValue;
            idata[i] /= denom;
        }
        return;
    }
#if unrolledIncrement == 0
    std::for_each(idata, idata + nSize, [&](T &d_i) { d_i = (d_i - MinValue) / denom; });
#else
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            idata[i] -= MinValue;
            idata[i] /= denom;
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
    {
        idata[i] -= MinValue;
        idata[i] /= denom;
        idata[i + 1] -= MinValue;
        idata[i + 1] /= denom;
        idata[i + 2] -= MinValue;
        idata[i + 2] /= denom;
        idata[i + 3] -= MinValue;
        idata[i + 3] /= denom;
#if unrolledIncrement == 6
        idata[i + 4] -= MinValue;
        idata[i + 4] /= denom;
        idata[i + 5] -= MinValue;
        idata[i + 5] /= denom;
#endif
#if unrolledIncrement == 8
        idata[i + 4] -= MinValue;
        idata[i + 4] /= denom;
        idata[i + 5] -= MinValue;
        idata[i + 5] /= denom;
        idata[i + 6] -= MinValue;
        idata[i + 6] /= denom;
        idata[i + 7] -= MinValue;
        idata[i + 7] /= denom;
#endif
#if unrolledIncrement == 10
        idata[i + 4] -= MinValue;
        idata[i + 4] /= denom;
        idata[i + 5] -= MinValue;
        idata[i + 5] /= denom;
        idata[i + 6] -= MinValue;
        idata[i + 6] /= denom;
        idata[i + 7] -= MinValue;
        idata[i + 7] /= denom;
        idata[i + 8] -= MinValue;
        idata[i + 8] /= denom;
        idata[i + 9] -= MinValue;
        idata[i + 9] /= denom;
#endif
#if unrolledIncrement == 12
        idata[i + 4] -= MinValue;
        idata[i + 4] /= denom;
        idata[i + 5] -= MinValue;
        idata[i + 5] /= denom;
        idata[i + 6] -= MinValue;
        idata[i + 6] /= denom;
        idata[i + 7] -= MinValue;
        idata[i + 7] /= denom;
        idata[i + 8] -= MinValue;
        idata[i + 8] /= denom;
        idata[i + 9] -= MinValue;
        idata[i + 9] /= denom;
        idata[i + 10] -= MinValue;
        idata[i + 10] /= denom;
        idata[i + 11] -= MinValue;
        idata[i + 11] /= denom;
#endif
    }
#endif
    return;
}

template <typename T>
void stats::minmaxNormal(std::vector<T> &idata)
{
    T const MinValue = minelement<T>(idata);
    T const MaxValue = maxelement<T>(idata);

    T denom = MaxValue - MinValue;
    if (denom < std::numeric_limits<T>::epsilon())
    {
        UMUQWARNING("Maximum and Minimum Value are identical!");
        denom = std::numeric_limits<T>::epsilon();
    }

    std::for_each(idata.begin(), idata.end(), [&](T &d_i) { d_i = (d_i - MinValue) / denom; });
    return;
}

template <typename T>
inline void stats::zscoreNormal(T *idata, int const nSize, int const Stride)
{
    T const MeanValue = mean<T, T>(idata, nSize, Stride);
    T const StddevValue = stddev<T, T>(idata, nSize, Stride, MeanValue);
    if (Stride > 1)
    {
        for (int i = 0; i < nSize; i += Stride)
        {
            idata[i] -= MeanValue;
            idata[i] /= StddevValue;
        }
        return;
    }
#if unrolledIncrement == 0
    std::for_each(idata, idata + nSize, [&](T &d_i) { d_i = (d_i - MeanValue) / StddevValue; });
#else
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            idata[i] -= MeanValue;
            idata[i] /= StddevValue;
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
    {
        idata[i] -= MeanValue;
        idata[i] /= StddevValue;
        idata[i + 1] -= MeanValue;
        idata[i + 1] /= StddevValue;
        idata[i + 2] -= MeanValue;
        idata[i + 2] /= StddevValue;
        idata[i + 3] -= MeanValue;
        idata[i + 3] /= StddevValue;
#if unrolledIncrement == 6
        idata[i + 4] -= MeanValue;
        idata[i + 4] /= StddevValue;
        idata[i + 5] -= MeanValue;
        idata[i + 5] /= StddevValue;
#endif
#if unrolledIncrement == 8
        idata[i + 4] -= MeanValue;
        idata[i + 4] /= StddevValue;
        idata[i + 5] -= MeanValue;
        idata[i + 5] /= StddevValue;
        idata[i + 6] -= MeanValue;
        idata[i + 6] /= StddevValue;
        idata[i + 7] -= MeanValue;
        idata[i + 7] /= StddevValue;
#endif
#if unrolledIncrement == 10
        idata[i + 4] -= MeanValue;
        idata[i + 4] /= StddevValue;
        idata[i + 5] -= MeanValue;
        idata[i + 5] /= StddevValue;
        idata[i + 6] -= MeanValue;
        idata[i + 6] /= StddevValue;
        idata[i + 7] -= MeanValue;
        idata[i + 7] /= StddevValue;
        idata[i + 8] -= MeanValue;
        idata[i + 8] /= StddevValue;
        idata[i + 9] -= MeanValue;
        idata[i + 9] /= StddevValue;
#endif
#if unrolledIncrement == 12
        idata[i + 4] -= MeanValue;
        idata[i + 4] /= StddevValue;
        idata[i + 5] -= MeanValue;
        idata[i + 5] /= StddevValue;
        idata[i + 6] -= MeanValue;
        idata[i + 6] /= StddevValue;
        idata[i + 7] -= MeanValue;
        idata[i + 7] /= StddevValue;
        idata[i + 8] -= MeanValue;
        idata[i + 8] /= StddevValue;
        idata[i + 9] -= MeanValue;
        idata[i + 9] /= StddevValue;
        idata[i + 10] -= MeanValue;
        idata[i + 10] /= StddevValue;
        idata[i + 11] -= MeanValue;
        idata[i + 11] /= StddevValue;
#endif
    }
#endif
    return;
}

template <typename T>
inline void stats::zscoreNormal(std::vector<T> &idata)
{
    T const MeanValue = mean<T, T>(idata);
    T const StddevValue = stddev<T, T>(idata, MeanValue);
    std::for_each(idata.begin(), idata.end(), [&](T &d_i) { d_i = (d_i - MeanValue) / StddevValue; });
    return;
}

template <typename T>
inline void stats::robustzscoreNormal(T *idata, int const nSize, int const Stride)
{
    T median_;
    T const mad = medianAbs<T, T>(idata, nSize, Stride, median_);
    if (Stride > 1)
    {
        for (int i = 0; i < nSize; i += Stride)
        {
            idata[i] -= median_;
            idata[i] /= mad;
        }
        return;
    }
#if unrolledIncrement == 0
    std::for_each(idata, idata + nSize, [&](T &d_i) { d_i = (d_i - median_) / mad; });
#else
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            idata[i] -= median_;
            idata[i] /= mad;
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
    {
        idata[i] -= median_;
        idata[i] /= mad;
        idata[i + 1] -= median_;
        idata[i + 1] /= mad;
        idata[i + 2] -= median_;
        idata[i + 2] /= mad;
        idata[i + 3] -= median_;
        idata[i + 3] /= mad;
#if unrolledIncrement == 6
        idata[i + 4] -= median_;
        idata[i + 4] /= mad;
        idata[i + 5] -= median_;
        idata[i + 5] /= mad;
#endif
#if unrolledIncrement == 8
        idata[i + 4] -= median_;
        idata[i + 4] /= mad;
        idata[i + 5] -= median_;
        idata[i + 5] /= mad;
        idata[i + 6] -= median_;
        idata[i + 6] /= mad;
        idata[i + 7] -= median_;
        idata[i + 7] /= mad;
#endif
#if unrolledIncrement == 10
        idata[i + 4] -= median_;
        idata[i + 4] /= mad;
        idata[i + 5] -= median_;
        idata[i + 5] /= mad;
        idata[i + 6] -= median_;
        idata[i + 6] /= mad;
        idata[i + 7] -= median_;
        idata[i + 7] /= mad;
        idata[i + 8] -= median_;
        idata[i + 8] /= mad;
        idata[i + 9] -= median_;
        idata[i + 9] /= mad;
#endif
#if unrolledIncrement == 12
        idata[i + 4] -= median_;
        idata[i + 4] /= mad;
        idata[i + 5] -= median_;
        idata[i + 5] /= mad;
        idata[i + 6] -= median_;
        idata[i + 6] /= mad;
        idata[i + 7] -= median_;
        idata[i + 7] /= mad;
        idata[i + 8] -= median_;
        idata[i + 8] /= mad;
        idata[i + 9] -= median_;
        idata[i + 9] /= mad;
        idata[i + 10] -= median_;
        idata[i + 10] /= mad;
        idata[i + 11] -= median_;
        idata[i + 11] /= mad;
#endif
    }
#endif
    return;
}

template <typename T>
inline void stats::robustzscoreNormal(std::vector<T> &idata)
{
    T median_;
    T const mad = medianAbs<T, T>(idata, median_);
    std::for_each(idata.begin(), idata.end(), [&](T &d_i) { d_i = (d_i - median_) / mad; });
    return;
}

template <typename T, typename TOut>
TOut stats::covariance(T const *idata, T const *jdata, int const nSize, T const iMean, T const jMean)
{
    /*!
     * \todo
     * If the data size is too big, maybe we should force long double
     */
    TOut Covariance(0);
    for (int i = 0; i < nSize; i++)
    {
        TOut const d1 = idata[i] - iMean;
        TOut const d2 = jdata[i] - jMean;
        Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(i + 1);
    }

    return (nSize > 1) ? Covariance * static_cast<TOut>(nSize) / static_cast<TOut>(nSize - 1) : Covariance;
}

template <typename T, typename TOut>
TOut stats::covariance(arrayWrapper<T> const &iArray, arrayWrapper<T> const &jArray, T const iMean, T const jMean)
{
    TOut Covariance(0);
    int iSize = 1;
    for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
    {
        TOut const d1 = *i - iMean;
        TOut const d2 = *j - jMean;

        Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(iSize++);
    }

    --iSize;

    return (iSize > 1) ? Covariance * static_cast<TOut>(iSize) / static_cast<TOut>(iSize - 1) : Covariance;
}

template <typename T, typename TOut>
TOut stats::covariance(std::vector<T> const &idata, std::vector<T> const &jdata, T const iMean, T const jMean)
{
    TOut Covariance(0);
    int iSize = 1;
    for (auto i = idata.begin(), j = jdata.begin(); i != idata.end(); i++, j++)
    {
        TOut const d1 = *i - iMean;
        TOut const d2 = *j - jMean;

        Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(iSize++);
    }

    --iSize;

    return (iSize > 1) ? Covariance * static_cast<TOut>(iSize) / static_cast<TOut>(iSize - 1) : Covariance;
}

template <typename T, typename TOut>
TOut stats::covariance(T const *idata, T const *jdata, int const nSize, int const Stride)
{
    T const iMean = mean<T, T>(idata, nSize, Stride);
    T const jMean = mean<T, T>(jdata, nSize, Stride);

    TOut Covariance(0);
    if (Stride != 1)
    {
        arrayWrapper<T> iArray(idata, nSize, Stride);
        arrayWrapper<T> jArray(jdata, nSize, Stride);

        int iSize = 1;
        for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
        {
            TOut const d1 = *i - iMean;
            TOut const d2 = *j - jMean;

            Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(iSize++);
        }

        --iSize;

        return (iSize > 1) ? Covariance * static_cast<TOut>(iSize) / static_cast<TOut>(iSize - 1) : Covariance;
    }

    for (int i = 0; i < nSize; i++)
    {
        TOut const d1 = idata[i] - iMean;
        TOut const d2 = jdata[i] - jMean;

        Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(i + 1);
    }

    return (nSize > 1) ? Covariance * static_cast<TOut>(nSize) / static_cast<TOut>(nSize - 1) : Covariance;
}

template <typename T, typename TOut>
TOut stats::covariance(arrayWrapper<T> const &iArray, arrayWrapper<T> const &jArray)
{
    T const iMean = mean<T, T>(iArray);
    T const jMean = mean<T, T>(jArray);

    TOut Covariance(0);
    int iSize = 1;
    for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
    {
        TOut const d1 = *i - iMean;
        TOut const d2 = *j - jMean;

        Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(iSize++);
    }

    --iSize;

    return (iSize > 1) ? Covariance * static_cast<TOut>(iSize) / static_cast<TOut>(iSize - 1) : Covariance;
}

template <typename T, typename TOut>
TOut stats::covariance(std::vector<T> const &idata, std::vector<T> const &jdata)
{
    T const iMean = mean<T, T>(idata);
    T const jMean = mean<T, T>(jdata);

    TOut Covariance(0);
    int iSize = 1;
    for (auto i = idata.begin(), j = jdata.begin(); i != idata.end(); i++, j++)
    {
        TOut const d1 = *i - iMean;
        TOut const d2 = *j - jMean;

        Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(iSize++);
    }

    --iSize;

    return (iSize > 1) ? Covariance * static_cast<TOut>(iSize) / static_cast<TOut>(iSize - 1) : Covariance;
}

template <typename T, typename TOut>
TOut *stats::covariance(T const *idata, int const nSize, int const nDim, int const Stride)
{
    TOut *Covariance = nullptr;
    try
    {
        Covariance = new TOut[nDim * nDim]();
    }
    catch (std::bad_alloc &e)
    {
        UMUQFAILRETURNNULL("Failed to allocate memory!");
    }

    std::vector<T> iMean(nDim);

    // We should make sure of the correct stride
    int const stride = Stride > nDim ? Stride : nDim;

    // Compute the mean for each dimension
    for (int i = 0; i < nDim; i++)
    {
        iMean[i] = mean<T, T>(idata + i, nSize, stride);
    }

    for (int i = 0; i < nDim; i++)
    {
        arrayWrapper<T> iArray(idata + i, nSize, stride);

        for (int j = i; j < nDim; j++)
        {
            arrayWrapper<T> jArray(idata + j, nSize, stride);

            Covariance[i * nDim + j] = covariance<T, TOut>(iArray, jArray, iMean[i], iMean[j]);
        }
    }

    for (int i = 0; i < nDim; i++)
    {
        for (int j = 0; j < i; j++)
        {
            Covariance[i * nDim + j] = Covariance[j * nDim + i];
        }
    }

    return Covariance;
}

template <typename T, typename TOut>
TOut *stats::covariance(T const *idata, int const nSize, int const nDim, T const *iMean)
{
    TOut *Covariance = nullptr;
    try
    {
        Covariance = new TOut[nDim * nDim]();
    }
    catch (std::bad_alloc &e)
    {
        UMUQFAILRETURNNULL("Failed to allocate memory!");
    }

    for (int i = 0; i < nDim; i++)
    {
        arrayWrapper<T> iArray(idata + i, nSize, nDim);

        for (int j = i; j < nDim; j++)
        {
            arrayWrapper<T> jArray(idata + j, nSize, nDim);

            Covariance[i * nDim + j] = covariance<T, TOut>(iArray, jArray, iMean[i], iMean[j]);
        }
    }

    for (int i = 0; i < nDim; i++)
    {
        for (int j = 0; j < i; j++)
        {
            Covariance[i * nDim + j] = Covariance[j * nDim + i];
        }
    }

    return Covariance;
}

template <typename T>
void stats::unique(T const *idata, int const nRows, int const nCols, std::vector<T> &udata)
{
    if (udata.size() < nRows * nCols)
    {
        // Resize the unique array to the maximum size
        udata.resize(nRows * nCols);
    }

    // Create a temporary array with the size of number of columns (one row of data)
    std::vector<T> x(nCols);

    // First element in the input array is considered unique
    std::copy(idata, idata + nCols, udata.begin());

    // We have one unique
    int nUniques(1);

    for (int i = 1; i < nRows; i++)
    {
        int const s = i * nCols;
        std::copy(idata + s, idata + s + nCols, x.begin());

        // Consider this x rows is unique among all the rows
        bool uniqueFlag(true);

        // check it with all the unique rows
        for (int j = 0, l = 0; j < nUniques; j++, l += nCols)
        {
            // Consider they are the same
            bool compareFlag = true;
            for (int k = 0; k < nCols; k++)
            {
                if (std::abs(x[k] - udata[l + k]) > 1e-6)
                {
                    // one element in the row differs, so they are different
                    compareFlag = false;
                    break;
                }
            }
            if (compareFlag)
            {
                // It is not a unique row
                uniqueFlag = false;
                break;
            }
        }

        if (uniqueFlag)
        {
            int const e = nUniques * nCols;
            std::copy(x.begin(), x.end(), udata.begin() + e);
            nUniques++;
        }
    }

    // Correct the size of the unique array
    if (nUniques * nCols < udata.size())
    {
        udata.resize(nUniques * nCols);
    }
    return;
}

template <typename T>
void stats::unique(std::vector<T> const &idata, int const nRows, int const nCols, std::vector<T> &udata)
{
    unique(idata.data(), nRows, nCols, udata);
}

template <typename T>
void stats::unique(std::unique_ptr<T[]> const &idata, int const nRows, int const nCols, std::vector<T> &udata)
{
    unique(idata.get(), nRows, nCols, udata);
}

} // namespace umuq

#endif // UMUQ_STATS
