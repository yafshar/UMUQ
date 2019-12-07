#ifndef UMUQ_STATS_H
#define UMUQ_STATS_H

#include "core/core.hpp"
#include "misc/arraywrapper.hpp"
#include "datatype/eigendatatype.hpp"
#include "eigenlib.hpp"

#include <cmath>

#include <vector>
#include <type_traits>
#include <limits>
#include <memory>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iterator>


namespace umuq
{

/*! \class stats
 * \ingroup Numerics_Module
 *
 * \brief stats is a class which includes some functionality for statistics of the input data
 *
 * It includes:<br>
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
 * - \b correlation        Compute the correlation coefficient (The population Pearson correlation coefficient.)
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
     * \tparam DataType Data type
     *
     * \param idata   Array of data
     * \param nSize   Size of the array
     * \param Stride  Element stride (default is 1)
     *
     * \returns DataType The smallest element in the array of data
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * int A[] = {2, 3, 5, 7, 1, 6, 8, 10, 9, 4, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
     *
     * std::cout << s.minelement<int>(A, 20) << std::endl;
     * std::cout << s.minelement<int>(A, 20, 2) << std::endl; // smallest element in A with stride = 2
     * std::cout << s.minelement<int>(A, 20, 5) << std::endl; // smallest element in A with stride = 5
     * \endcode
     *
     * Output:<br>
     * \code
     * -10
     * -9
     * -6
     * \endcode
     */
    template <typename DataType>
    inline DataType minelement(DataType const *idata, int const nSize, int const Stride = 1) const;

    /*!
     * \brief Finds the smallest element in the array of data with stride
     *
     * \tparam DataType Data type
     *
     * \param idata  Array of data
     *
     * \returns DataType The smallest element in the array of data
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * std::vector<double> B{2, 3, 5, 7, 1, 6, 8, 10, 9, 4};
     *
     * std::cout << s.minelement<double>(B) << std::endl;
     * \endcode
     *
     * Output:<br>
     * \code
     * 1
     * \endcode
     */
    template <typename DataType>
    inline DataType minelement(std::vector<DataType> const &idata) const;

    /*!
     * \brief Finds the smallest element in the array of data with stride
     *
     * \tparam DataType Data type
     *
     * \param iArray  Array of data
     *
     * \returns DataType The smallest element in the array of data
     */
    template <typename DataType>
    inline DataType minelement(arrayWrapper<DataType> const &iArray) const;

    /*!
     * \brief Finds the greatest element in the array of data with stride
     *
     * \tparam DataType Data type
     *
     * \param idata   Array of data
     * \param nSize   Size of the array
     * \param Stride  Element stride (default is 1)
     *
     * \returns DataType The greatest element in the array of data
     */
    template <typename DataType>
    inline DataType maxelement(DataType const *idata, int const nSize, int const Stride = 1) const;

    /*!
     * \brief Finds the greatest element in the array of data with stride
     *
     * \tparam DataType Data type
     *
     * \param idata   Array of data
     *
     * \returns DataType The greatest element in the array of data
     */
    template <typename DataType>
    inline DataType maxelement(std::vector<DataType> const &idata) const;

    /*!
     * \brief Finds the greatest element in the array of data with stride
     *
     * \tparam DataType Data type
     *
     * \param iArray Array of data
     *
     * \returns DataType The greatest element in the array of data
     */
    template <typename DataType>
    inline DataType maxelement(arrayWrapper<DataType> const &iArray) const;

    /*!
     * \brief Finds the position of the smallest element in the array of data (idata) with stride
     *
     * \tparam DataType Data type
     *
     * \param idata   Array of data
     * \param nSize   Size of the array
     * \param Stride  Element stride (default is 1)
     *
     * \returns int The position of the smallest element
     */
    template <typename DataType>
    inline int minelement_index(DataType const *idata, int const nSize, int const Stride = 1) const;

    /*!
     * \brief Finds the position of the smallest element in the array of data (idata) with stride
     *
     * \tparam DataType Data type
     *
     * \param idata  Array of data
     *
     * \returns int The position of the smallest element
     */
    template <typename DataType>
    inline int minelement_index(std::vector<DataType> const &idata) const;

    /*!
     * \brief Finds the position of the smallest element in the array of data (idata) with stride
     *
     * \tparam DataType Data type
     *
     * \param iArray  Array of data
     *
     * \returns int The position of the smallest element
     */
    template <typename DataType>
    inline int minelement_index(arrayWrapper<DataType> const &iArray) const;

    /*!
     * \brief Finds the position of the greatest element in the array of data (idata) with Stride
     *
     * \tparam DataType Data type
     *
     * \param idata   Array of data
     * \param nSize   Size of the array
     * \param Stride  Element stride (default is 1)
     *
     * \returns int The the position of the greatest element
     */
    template <typename DataType>
    inline int maxelement_index(DataType const *idata, int const nSize, int const Stride = 1) const;

    /*!
     * \brief Finds the position of the greatest element in the array of data (idata) with Stride
     *
     * \tparam DataType Data type
     *
     * \param idata  Array of data
     *
     * \returns int The the position of the greatest element
     */
    template <typename DataType>
    inline int maxelement_index(std::vector<DataType> const &idata) const;

    /*!
     * \brief Finds the position of the greatest element in the array of data (idata) with Stride
     *
     * \tparam DataType Data type
     *
     * \param iArray  Array of data
     *
     * \returns int The the position of the greatest element
     */
    template <typename DataType>
    inline int maxelement_index(arrayWrapper<DataType> const &iArray) const;

    /*!
     * \brief Computes the sum of the elements in the array of data with stride
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param nSize   Size of the array
     * \param Stride  Element stride (default is 1)
     *
     * \returns OutputDataType The sum of the elements in the array of data
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType sum(DataType const *idata, int const nSize, int const Stride = 1) const;

    /*!
     * \brief Computes the sum of the elements in the array of data with stride
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param idata  Array of data
     *
     * \returns OutputDataType The sum of the elements in the array of data
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType sum(std::vector<DataType> const &idata) const;

    /*!
     * \brief Computes the sum of the elements in the array of data with stride
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param iArray  Array of data
     *
     * \returns OutputDataType The sum of the elements in the array of data
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType sum(arrayWrapper<DataType> const &iArray) const;

    /*!
     * \brief Computes the sum of the absolute value of the elements in the array of data with stride
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param nSize   Size of the array
     * \param Stride  Element stride (default is 1)
     *
     * \returns OutputDataType The sum of the absolute value of the elements in the array of data
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType sumAbs(DataType const *idata, int const nSize, int const Stride = 1) const;

    /*!
     * \brief Computes the sum of the absolute value of the elements in the array of data with stride
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param idata  Array of data
     *
     * \returns OutputDataType The sum of the absolute value of the elements in the array of data
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType sumAbs(std::vector<DataType> const &idata) const;

    /*!
     * \brief Computes the sum of the absolute value of the elements in the array of data with stride
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param iArray  Array of data
     *
     * \returns OutputDataType The sum of the absolute value of the elements in the array of data
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType sumAbs(arrayWrapper<DataType> const &iArray) const;

    /*!
     * \brief Computes the mean of the elements in the array of data with stride
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param nSize   Size of the array
     * \param Stride  Element stride (default is 1)
     *
     * \returns RealType The mean of the elements in the array of data
     */
    template <typename DataType, typename RealType = double>
    inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    mean(DataType const *idata, int const nSize, int const Stride = 1) const;

    /*!
     * \brief Computes the mean of the elements in the array of data with stride
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata   Array of data
     *
     * \returns RealType The mean of the elements in the array of data
     */
    template <typename DataType, typename RealType = double>
    inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    mean(std::vector<DataType> const &idata) const;

    /*!
     * \brief Computes the mean of the elements in the array of data with stride
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param iArray  Array of data
     *
     * \returns RealType The mean of the elements in the array of data
     */
    template <typename DataType, typename RealType = double>
    inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    mean(arrayWrapper<DataType> const &iArray) const;

    /*!
     * \brief Computes the median of the elements in the array of data with Stride
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param nSize   Size of the array
     * \param Stride  Element stride (default is 1)
     *
     * \returns OutputDataType The median of the elements in the array of data with Stride
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType median(DataType const *idata, int const nSize, int const Stride = 1);

    /*!
     * \brief Computes the median of the elements in the array of data with Stride
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param idata  Array of data
     *
     * \returns OutputDataType The median of the elements in the array of data with Stride
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType median(std::vector<DataType> const &idata);

    /*!
     * \brief Computes the median of the elements in the array of data with Stride
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param iArray  Array of data
     *
     * \returns OutputDataType The median of the elements in the array of data with Stride
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType median(arrayWrapper<DataType> const &iArray);

    /*!
     * \brief Computes the median absolute deviation (MAD) of the elements in the array of data
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param nSize   Size of the array
     * \param Stride  Element stride
     * \param Median  Median of the elements in the array of data
     *
     * \returns OutputDataType The median absolute deviation of the elements in the array of data
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType medianAbs(DataType const *idata, int const nSize, int const Stride = 1, OutputDataType &Median = OutputDataType{});

    /*!
     * \brief Computes the median absolute deviation (MAD) of the elements in the array of data
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param Median  Median of the elements in the array of data
     *
     * \returns OutputDataType The median absolute deviation of the elements in the array of data
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType medianAbs(std::vector<DataType> const &idata, OutputDataType &Median = OutputDataType{});

    /*!
     * \brief Computes the median absolute deviation (MAD) of the elements in the array of data
     *
     * \tparam DataType       Data type
     * \tparam OutputDataType Data type of the return output result (default is double)
     *
     * \param iArray  Array of data
     * \param Median  Median of the elements in the array of data
     *
     * \returns OutputDataType The median absolute deviation of the elements in the array of data
     */
    template <typename DataType, typename OutputDataType = double>
    inline OutputDataType medianAbs(arrayWrapper<DataType> const &iArray, OutputDataType &Median = OutputDataType{});

    /*!
     * \brief Computes the standard deviation of the elements in the array of data with or without stride
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata      Array of data
     * \param nSize      Size of the array
     * \param Stride     Element stride (optional, default is 1)
     * \param idataMean  Mean of the elements in idata (optional)
     *
     * \returns RealType The standard deviation of the elements in the array of data
     */
    template <typename DataType, typename RealType = double>
    inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    stddev(DataType const *idata, int const nSize, int const Stride = 1, RealType const idataMean = std::numeric_limits<RealType>::max()) const;

    /*!
     * \brief Computes the standard deviation of the elements in the array of data with or without stride
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata      Array of data
     * \param idataMean  Mean of the elements in idata (optional)
     *
     * \returns RealType The standard deviation of the elements in the array of data
     */
    template <typename DataType, typename RealType = double>
    inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    stddev(std::vector<DataType> const &idata, RealType const idataMean = std::numeric_limits<RealType>::max()) const;
    /*!
     * \brief Computes the standard deviation of the elements in the array of data with or without stride
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param iArray     Array of data
     * \param idataMean  Mean of the elements in idata (optional)
     *
     * \returns RealType The standard deviation of the elements in the array of data
     */
    template <typename DataType, typename RealType = double>
    inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    stddev(arrayWrapper<DataType> const &iArray, RealType const idataMean = std::numeric_limits<RealType>::max()) const;

    /*!
     * \brief Computes the coefficient of variation (CV), or relative standard deviation (RSD).
     * It is a standardized measure of dispersion of a probability distribution or frequency distribution.
     * It is defined as the ratio of the standard deviation \f$ \sigma \f$ to the mean \f$ \mu \f$
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata      Array of data
     * \param nSize      Size of the array
     * \param Stride     Element stride (optional, default is 1)
     * \param idataMean  Mean of the elements in idata (optional)
     *
     * \returns RealType The coefficient of variation (CV)
     */
    template <typename DataType, typename RealType = double>
    inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    coefvar(DataType const *idata, int const nSize, int const Stride = 1, RealType const idataMean = std::numeric_limits<RealType>::max()) const;

    /*!
     * \brief Computes the coefficient of variation (CV), or relative standard deviation (RSD).
     * It is a standardized measure of dispersion of a probability distribution or frequency distribution.
     * It is defined as the ratio of the standard deviation \f$ \sigma \f$ to the mean \f$ \mu \f$
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata      Array of data
     * \param idataMean  Mean of the elements in idata (optional)
     *
     * \returns RealType The coefficient of variation (CV)
     */
    template <typename DataType, typename RealType = double>
    inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    coefvar(std::vector<DataType> const &idata, RealType const idataMean = std::numeric_limits<RealType>::max()) const;

    /*!
     * \brief Computes the coefficient of variation (CV), or relative standard deviation (RSD).
     * It is a standardized measure of dispersion of a probability distribution or frequency distribution.
     * It is defined as the ratio of the standard deviation \f$ \sigma \f$ to the mean \f$ \mu \f$
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param iArray     Array of data
     * \param idataMean  Mean of the elements in idata (optional)
     *
     * \returns RealType The coefficient of variation (CV)
     */
    template <typename DataType, typename RealType = double>
    inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    coefvar(arrayWrapper<DataType> const &iArray, RealType const idataMean = std::numeric_limits<RealType>::max()) const;

    /*!
     * \brief minmaxNormal scales the numeric data using the MinMax normalization method
     *
     * Using the MinMax normalization method, one can normalize the values to be between 0 and 1.
     * Doing so allows to compare values on very different scales to one another by reducing
     * the dominance of one dimension over the other.
     *
     * \tparam DataType Data type
     *
     * \param idata     Array of data
     * \param nSize     Size of array
     * \param Stride    Element stride (default is 1)
     * \param MinValue  Input minimum value
     * \param MaxValue  Input maximum value
     */
    template <typename DataType>
    void minmaxNormal(DataType *idata, int const nSize, int const Stride = 1, DataType const MinValue = UFAIL, DataType const MaxValue = UFAIL);

    /*!
     * \brief minmaxNormal scales the numeric data using the MinMax normalization method
     *
     * Using the MinMax normalization method, one can normalize the values to be between 0 and 1.
     * Doing so allows to compare values on very different scales to one another by reducing
     * the dominance of one dimension over the other.
     *
     * \tparam DataType Data type
     *
     * \param idata     Array of data
     * \param nSize     Size of array
     * \param MinValue  Output minimum value
     * \param MaxValue  Output maximum value
     * \param Stride    Element stride (default is 1)
     */
    template <typename DataType>
    void minmaxNormal(DataType *idata, int const nSize, DataType &MinValue, DataType &MaxValue, int const Stride = 1);

    /*!
     * \brief minmaxNormal scales the numeric data using the MinMax normalization method
     *
     * Using the MinMax normalization method, one can normalize the values to be between 0 and 1.
     * Doing so allows to compare values on very different scales to one another by reducing
     * the dominance of one dimension over the other.
     *
     * \tparam DataType Data type
     *
     * \param idata     Array of data
     * \param MinValue  Input minimum value
     * \param MaxValue  Input maximum value
     */
    template <typename DataType>
    void minmaxNormal(std::vector<DataType> &idata, DataType const MinValue = UFAIL, DataType const MaxValue = UFAIL);

    /*!
     * \brief minmaxNormal scales the numeric data using the MinMax normalization method
     *
     * Using the MinMax normalization method, one can normalize the values to be between 0 and 1.
     * Doing so allows to compare values on very different scales to one another by reducing
     * the dominance of one dimension over the other.
     *
     * \tparam DataType Data type
     *
     * \param idata     Array of data
     * \param MinValue  Output minimum value
     * \param MaxValue  Output maximum value
     */
    template <typename DataType>
    void minmaxNormal(std::vector<DataType> &idata, DataType &MinValue, DataType &MaxValue);

    /*!
     * \brief zscoreNormal scales the numeric data using the Z-score normalization method
     *
     * Using the Z-score normalization method, one can normalize the values to be the number of
     * standard deviations an observation is from the mean of each dimension.
     * This allows to compare data to a normally distributed random variable.
     *
     * \tparam DataType Data type
     *
     * \param idata  Input data
     * \param nSize  Size of array
     * \param Stride Element stride (default is 1)
     */
    template <typename DataType>
    inline void zscoreNormal(DataType *idata, int const nSize, int const Stride = 1);

    /*!
     * \brief zscoreNormal scales the numeric data using the Z-score normalization method
     *
     * Using the Z-score normalization method, one can normalize the values to be the number of
     * standard deviations an observation is from the mean of each dimension.
     * This allows to compare data to a normally distributed random variable.
     *
     * \tparam DataType Data type
     *
     * \param idata  Input data
     */
    template <typename DataType>
    inline void zscoreNormal(std::vector<DataType> &idata);

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
     * \tparam DataType Data type
     *
     * \param idata  Input data
     * \param nSize  Size of the array
     * \param Stride Element stride (default is 1)
     */
    template <typename DataType>
    inline void robustzscoreNormal(DataType *idata, int const nSize, int const Stride = 1);

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
     * \tparam DataType Data type
     *
     * \param idata  Input data
     */
    template <typename DataType>
    inline void robustzscoreNormal(std::vector<DataType> &idata);

    /*!
     * \brief Compute the covariance between idata and jdata vectors which must both be of the same length nSize <br>
     * \f$ \text{covariance}(idata, jdata) = \frac{1}{n-1} \sum_{i=1}^n (idata_i-iMean)(jdata_i-jMean) \f$
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata  Array of data
     * \param jdata  Array of data
     * \param nSize  Size of array
     * \param iMean  Mean of idata array
     * \param jMean  Mean of jdata array
     *
     * \returns Covariance (scaler value) between idata and jdata vectors
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * double A[] = {2.1, 2.5, 3.6, 4.0}; // (mean = 3.1)
     * double B[] = {8, 10, 12, 14};	  // (mean = 11)
     *
     * std::cout << s.covariance<double, double>(A, B, 4, 3.1, 11) << std::endl;
     * \endcode
     *
     * Output:<br>
     * \code
     * 2.26667
     * \endcode
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    covariance(DataType const *idata, DataType const *jdata, int const nSize, RealType const iMean, RealType const jMean);

    /*!
     * \brief Compute the covariance between two arrays of data which must both be of the same length
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param iArray  Array of data
     * \param jArray  Array of data
     * \param iMean   Mean of iArray
     * \param jMean   Mean of jArray
     *
     * \returns Covariance (scaler value) between idata and jdata vectors
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    covariance(arrayWrapper<DataType> const &iArray, arrayWrapper<DataType> const &jArray, RealType const iMean, RealType const jMean);

    /*!
     * \brief Compute the covariance between two arrays of data which must both be of the same length
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param jdata   Array of data
     * \param iMean   Mean of iArray
     * \param jMean   Mean of jArray
     *
     * \returns Covariance (scaler value) between idata and jdata vectors
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * std::vector<double> A{2.1, 2.5, 3.6, 4.0}; // (mean = 3.1)
     * std::vector<double> B{8, 10, 12, 14};	  // (mean = 11)
     *
     * std::cout << s.covariance<double, double>(A, B, 3.1, 11) << std::endl;
     * \endcode
     *
     * Output:<br>
     * \code
     * 2.26667
     * \endcode
     *
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    covariance(std::vector<DataType> const &idata, std::vector<DataType> const &jdata, RealType const iMean, RealType const jMean);

    /*!
     * \brief Compute the covariance between idata and jdata vectors which must both be of the same length \c nSize
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param jdata   Array of data
     * \param nSize   Size of array
     * \param Stride  Stride of the data in the array (default is 1)
     *
     * \returns Covariance (scaler value) between idata and jdata vectors
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * double A[] = {2.1, 2.5, 3.6, 4.0}; // (with stride 2 {2.1, 3.6} - mean = 2.85)
     * double B[] = {8, 10, 12, 14};	  // (with stride 2 {8, 12}    - mean = 10)
     *
     * std::cout << s.covariance<double, double>(A, B, 4, 2) << std::endl;
     * \endcode
     *
     * Output:<br>
     * \code
     * 3
     * \endcode
     *
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    covariance(DataType const *idata, DataType const *jdata, int const nSize, int const Stride = 1);

    /*!
     * \brief Compute the covariance between two arrays of data which must both be of the same length
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param iArray  Array of data
     * \param jArray  Array of data
     *
     * \returns Covariance (scaler value) between idata and jdata vectors
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    covariance(arrayWrapper<DataType> const &iArray, arrayWrapper<DataType> const &jArray);

    /*!
     * \brief Compute the covariance between two arrays of data which must both be of the same length
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata  Array of data
     * \param jdata  Array of data
     *
     * \returns Covariance (scaler value) between idata and jdata vectors
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * std::vector<double> A{2.1, 3.6}; // (mean = 2.85)
     * std::vector<double> B{8, 12};    // (mean = 10)
     *
     * std::cout << s.covariance<double, double>(A, B) << std::endl;
     * \endcode
     *
     * Output:<br>
     * \code
     * 3
     * \endcode
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    covariance(std::vector<DataType> const &idata, std::vector<DataType> const &jdata);

    /*!
     * \brief Compute the covariance array of N-dimensional idata
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
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
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * // A 3-by-4 matrix
     * double A[] = {5,  0, 3, 7,
     *               1, -5, 7, 3,
     *               4,  9, 8, 10};
     *
     * double *Covariance = s.covariance<double, double>(A, 12, 4, 4);
     *
     * for (int i = 0, l = 0; i < 4; i++)
     * {
     *      for (int j = 0; j < 4; j++)
     *          std::cout << Covariance[l++] << " ";
     *      std::cout << std::endl;
     * }
     * \endcode
     *
     * Output:<br>
     * \f$
     * \begin{matrix}
     * 4.33333 & 8.83333 & -3 & 5.66667 \\
     * 8.83333 & 50.3333 & 6.5 & 24.1667  \\
     * -3 & 6.5 & 7 & 1 \\
     * 5.66667 & 24.1667 & 1 & 12.3333
     * \end{matrix}
     * \f$
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
        *covariance(DataType const *idata, int const nSize, int const nDim, int const Stride = 1);

    /*!
     * \brief Compute the covariance array of N-dimensional idata
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata  Array of N-dimensional data with size of [nSize/nDim][nDim]
     * \param nSize  Total size of the array
     * \param nDim   Data dimension
     * \param iMean  Mean of each column or dimension of the array idata with size of [nDim]
     *
     * \returns Covariance (array of [nDim * nDim]) from N-dimensional idata
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
        *covariance(DataType const *idata, int const nSize, int const nDim, RealType const *iMean);

    /*!
     * \brief Compute the correlation between idata and jdata vectors which must both be of the same length nSize <br>
     * Correlation coefficient when applied to a population is commonly represented by the Greek letter \f$ \rho \f$
     * and may be referred to as the population correlation coefficient or the population Pearson correlation coefficient.
     * \f$ \rho (idata, jdata) = correlation (idata, jdata) = \frac{covariance (idata, jdata) }{stddev(idata) stddev(jdata) } \f$
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata  Array of data
     * \param jdata  Array of data
     * \param nSize  Size of array
     * \param iMean  Mean of idata array
     * \param jMean  Mean of jdata array
     *
     * \returns Correlation (scaler value) between idata and jdata vectors
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * double X[] = {2.1, 2.5, 3.6, 4.0}; // (mean = 3.1)
     * double Y[] = {8, 10, 12, 14};	  // (mean = 11)
     *
     * std::cout << s.correlation<double, double>(X, Y, 4, 3.1, 11) << std::endl;
     * \endcode
     *
     * Output:<br>
     * \code
     * 0.977431
     * \endcode
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    correlation(DataType const *idata, DataType const *jdata, int const nSize, RealType const iMean, RealType const jMean);

    /*!
     * \brief Compute the correlation between two arrays of data which must both be of the same length
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param iArray  Array of data
     * \param jArray  Array of data
     * \param iMean   Mean of iArray
     * \param jMean   Mean of jArray
     *
     * \returns Correlation (scaler value) between idata and jdata vectors
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    correlation(arrayWrapper<DataType> const &iArray, arrayWrapper<DataType> const &jArray, RealType const iMean, RealType const jMean);

    /*!
     * \brief Compute the correlation between two arrays of data which must both be of the same length
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param jdata   Array of data
     * \param iMean   Mean of iArray
     * \param jMean   Mean of jArray
     *
     * \returns Correlation (scaler value) between idata and jdata vectors
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * std::vector<double> X{2.1, 2.5, 3.6, 4.0}; // (mean = 3.1)
     * std::vector<double> Y{8, 10, 12, 14};	  // (mean = 11)
     *
     * std::cout << s.correlation<double, double>(X, Y, 3.1, 11) << std::endl;
     * \endcode
     *
     * Output:<br>
     * \code
     * 0.977431
     * \endcode
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    correlation(std::vector<DataType> const &idata, std::vector<DataType> const &jdata, RealType const iMean, RealType const jMean);

    /*!
     * \brief Compute the correlation between idata and jdata vectors which must both be of the same length \c nSize
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata   Array of data
     * \param jdata   Array of data
     * \param nSize   Size of array
     * \param Stride  Stride of the data in the array (default is 1)
     *
     * \returns Correlation (scaler value) between idata and jdata vectors
     *
     * Compute the correlation between idata and jdata vectors which must both be of the same length nSize <br>
     * Correlation coefficient when applied to a population is commonly represented by the Greek letter \f$ \rho \f$
     * and may be referred to as the population correlation coefficient or the population Pearson correlation coefficient.
     * \f$ \rho (idata, jdata) = correlation (idata, jdata) = \frac{covariance (idata, jdata) }{stddev(idata) stddev(jdata) } \f$
     * It computes the correlation in one pass of the data and makes use of the algorithm described in Welford~\cite{Welford1962},
     * where it uses a numerically stable recurrence to compute a sum of products:<br>
     * \f$ S = \sum_{i=1}^{nSize} {[(idata_i - \overline{idata}) \times (jdata_i - \overline{jdata})  ]}, \f$ <br>
     * with the relation <br>
     * \f$ S_n = S_{n-1} + ((n-1)/n) * (idata_n - \overline{idata}_{n-1}) * (jdata_n - \overline{jdata}_{n-1}). \f$
     *
     * Reference:<br>
     * B. P. Welford, "Note on a Method for Calculating Corrected Sums of
     * Squares and Products", Technometrics, Vol 4, No 3, 1962.
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * double X[] = {2.1, 2.5, 3.6, 4.0};
     * double Y[] = {8, 10, 12, 14};
     *
     * std::cout << s.correlation<double, double>(X, Y, 4) << std::endl;
     * \endcode
     *
     * Output:<br>
     * \code
     * 0.979457
     * \endcode
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    correlation(DataType const *idata, DataType const *jdata, int const nSize, int const Stride = 1);

    /*!
     * \brief Compute the correlation between two arrays of data which must both be of the same length
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param iArray  Array of data
     * \param jArray  Array of data
     *
     * \returns Correlation (scaler value) between iArray and jArray vectors
     *
     * Compute the correlation between iArray and jArray vectors which must both be of the same length nSize <br>
     * Correlation coefficient when applied to a population is commonly represented by the Greek letter \f$ \rho \f$
     * and may be referred to as the population correlation coefficient or the population Pearson correlation coefficient.
     * \f$ \rho (iArray, jArray) = correlation (iArray, jArray) = \frac{covariance (iArray, jArray) }{stddev(iArray) stddev(jArray) } \f$
     * It computes the correlation in one pass of the data and makes use of the algorithm described in Welford~\cite{Welford1962},
     * where it uses a numerically stable recurrence to compute a sum of products:<br>
     * \f$ S = \sum_{i=1}^{nSize} {[(idata_i - \overline{iArray}) \times (jdata_i - \overline{jArray})  ]}, \f$ <br>
     * with the relation <br>
     * \f$ S_n = S_{n-1} + ((n-1)/n) * (idata_n - \overline{iArray}_{n-1}) * (jdata_n - \overline{jArray}_{n-1}). \f$
     *
     * Reference:<br>
     * B. P. Welford, "Note on a Method for Calculating Corrected Sums of
     * Squares and Products", Technometrics, Vol 4, No 3, 1962.
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    correlation(arrayWrapper<DataType> const &iArray, arrayWrapper<DataType> const &jArray);

    /*!
     * \brief Compute the correlation between two arrays of data which must both be of the same length
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata  Array of data
     * \param jdata  Array of data
     *
     * \returns Correlation (scaler value) between idata and jdata vectors
     *
     * Compute the correlation between idata and jdata vectors which must both be of the same length nSize <br>
     * Correlation coefficient when applied to a population is commonly represented by the Greek letter \f$ \rho \f$
     * and may be referred to as the population correlation coefficient or the population Pearson correlation coefficient.
     * \f$ \rho (idata, jdata) = correlation (idata, jdata) = \frac{covariance (idata, jdata) }{stddev(idata) stddev(jdata) } \f$
     * It computes the correlation in one pass of the data and makes use of the algorithm described in Welford~\cite{Welford1962},
     * where it uses a numerically stable recurrence to compute a sum of products:<br>
     * \f$ S = \sum_{i=1}^{nSize} {[(idata_i - \overline{idata}) \times (jdata_i - \overline{jdata})  ]}, \f$ <br>
     * with the relation <br>
     * \f$ S_n = S_{n-1} + ((n-1)/n) * (idata_n - \overline{idata}_{n-1}) * (jdata_n - \overline{jdata}_{n-1}). \f$
     *
     * Reference:<br>
     * B. P. Welford, "Note on a Method for Calculating Corrected Sums of
     * Squares and Products", Technometrics, Vol 4, No 3, 1962.
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * std::vector<int> X{17, 18, 16, 18, 12, 20, 18, 20, 20, 22, 20, 10, 8, 12, 16, 16, 18, 20, 18, 21};
     * std::vector<int> Y{19, 20, 22, 24, 10, 25, 20, 22, 21, 23, 20, 10, 12, 14, 12, 20, 22, 24, 23, 17};
     *
     * std::cout << s.correlation<int, double>(X, Y) << std::endl;
     * \endcode
     *
     * Output:<br>
     * \code
     * 0.79309
     * \endcode
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    correlation(std::vector<DataType> const &idata, std::vector<DataType> const &jdata);

    /*!
     * \brief Compute the correlation array of N-dimensional idata
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata  Array of N-dimensional data
     * \param nSize  Total size of the array
     * \param nDim   Data dimension
     * \param Stride Stride of the data in the array (default is 1).
     *
     * The reason for having parameter stride is the case where we have coordinates and function value
     * and would like to avoid unnecessary copying the data
     *
     * \returns Correlation (array of N by N) from N-dimensional idata
     *
     * Example:<br>
     * \code
     * umuq::stats s;
     *
     * // X 3-by-4 matrix
     * double X[] = {5,  0, 3, 7,
     *               1, -5, 7, 3,
     *               4,  9, 8, 10};
     *
     * double *Correlation = s.correlation<double, double>(X, 12, 4, 4);
     *
     * for (int i = 0, l = 0; i < 4; i++)
     * {
     *      for (int j = 0; j < 4; j++)
     *          std::cout << Correlation[l++] << " ";
     *      std::cout << std::endl;
     * }
     * \endcode
     *
     * Output:<br>
     * \f$
     * \begin{matrix}
     * 1 & 0.598116 & -0.544705 & 0.775133 \\
     * 0.598116 & 1 & 0.346287 & 0.969948 \\
     * -0.544705 & 0.346287 & 1 & 0.107624 \\
     * 0.775133 & 0.969948 & 0.107624  & 1
     * \end{matrix}
     * \f$
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
        *correlation(DataType const *idata, int const nSize, int const nDim, int const Stride = 1);

    /*!
     * \brief Compute the correlation array of N-dimensional idata
     *
     * \tparam DataType Data type
     * \tparam RealType Floating point data type of the return output result (default is double)
     *
     * \param idata  Array of N-dimensional data with size of [nSize/nDim][nDim]
     * \param nSize  Total size of the array
     * \param nDim   Data dimension
     * \param iMean  Mean of each column or dimension of the array idata with size of [nDim]
     *
     * \returns Correlation (array of [nDim * nDim]) from N-dimensional idata
     */
    template <typename DataType, typename RealType = double>
    std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
        *correlation(DataType const *idata, int const nSize, int const nDim, RealType const *iMean);

    /*!
     * \brief Eliminates all but the first element from every consecutive sample points of dimension n = nCols
     * of equivalent elements from idata which is an array of size nRows * nCols.
     * Find the unique n-dimensions sample points in an array of nRows * nCols data.
     *
     * \tparam DataType Data type
     *
     * \param idata  Input data
     * \param nRows  Number of rows
     * \param nCols  Number of columns (data dimension)
     * \param udata  Unique data (every row in this data is unique)
     */
    template <typename DataType>
    void unique(DataType const *idata, int const nRows, int const nCols, std::vector<DataType> &udata);

    /*!
     * \brief Eliminates all but the first element from every consecutive sample points of dimension n = nCols
     * of equivalent elements from idata which is an array of size nRows * nCols.
     * Find the unique n-dimensions sample points in an array of nRows * nCols data.
     *
     * \tparam DataType Data type
     *
     * \param idata  Input data
     * \param nRows  Number of rows
     * \param nCols  Number of columns (data dimension)
     * \param udata  Unique data (every row in this data is unique)
     */
    template <typename DataType>
    void unique(std::vector<DataType> const &idata, int const nRows, int const nCols, std::vector<DataType> &udata);

    /*!
     * \brief Eliminates all but the first element from every consecutive sample points of dimension n = nCols
     * of equivalent elements from idata which is an array of size nRows * nCols.
     * Find the unique n-dimensions sample points in an array of nRows * nCols data.
     *
     * \tparam DataType Data type
     *
     * \param idata  Input data
     * \param nRows  Number of rows
     * \param nCols  Number of columns (data dimension)
     * \param udata  Unique data (every row in this data is unique)
     */
    template <typename DataType>
    void unique(std::unique_ptr<DataType[]> const &idata, int const nRows, int const nCols, std::vector<DataType> &udata);
};

stats::stats()
{
    if (!(unrolledIncrement == 0 || unrolledIncrement == 4 || unrolledIncrement == 6 || unrolledIncrement == 8 || unrolledIncrement == 10 || unrolledIncrement == 12))
    {
        UMUQFAIL("The unrolled increment value of unrolledIncrement is not correctly set!");
    }
}

stats::~stats() {}

template <typename DataType>
inline DataType stats::minelement(DataType const *idata, int const nSize, int const Stride) const
{
    if (Stride > 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? *std::min_element(iArray.begin(), iArray.end()) : throw(std::runtime_error("Wrong input size!"));
    }
    return (nSize > 0) ? *std::min_element(idata, idata + nSize) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline DataType stats::minelement(std::vector<DataType> const &idata) const
{
    return (idata.size() > 0) ? *std::min_element(idata.begin(), idata.end()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline DataType stats::minelement(arrayWrapper<DataType> const &iArray) const
{
    return (iArray.size() > 0) ? *std::min_element(iArray.begin(), iArray.end()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline DataType stats::maxelement(DataType const *idata, int const nSize, int const Stride) const
{
    if (Stride > 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? *std::max_element(iArray.begin(), iArray.end()) : throw(std::runtime_error("Wrong input size!"));
    }
    return (nSize > 0) ? *std::max_element(idata, idata + nSize) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline DataType stats::maxelement(std::vector<DataType> const &idata) const
{
    return (idata.size() > 0) ? *std::max_element(idata.begin(), idata.end()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline DataType stats::maxelement(arrayWrapper<DataType> const &iArray) const
{
    return (iArray.size() > 0) ? *std::max_element(iArray.begin(), iArray.end()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline int stats::minelement_index(DataType const *idata, int const nSize, int const Stride) const
{
    if (Stride > 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? static_cast<int>(std::distance(iArray.begin(), std::min_element(iArray.begin(), iArray.end())) * Stride) : throw(std::runtime_error("Wrong input size!"));
    }

    return (nSize > 0) ? static_cast<int>(std::distance(idata, std::min_element(idata, idata + nSize))) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline int stats::minelement_index(std::vector<DataType> const &idata) const
{
    return (idata.size() > 0) ? static_cast<int>(std::distance(idata.begin(), std::min_element(idata.begin(), idata.end()))) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline int stats::minelement_index(arrayWrapper<DataType> const &iArray) const
{
    return (iArray.size() > 0) ? static_cast<int>(std::distance(iArray.begin(), std::min_element(iArray.begin(), iArray.end())) * iArray.stride()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline int stats::maxelement_index(DataType const *idata, int const nSize, int const Stride) const
{
    if (Stride > 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? static_cast<int>(std::distance(iArray.begin(), std::max_element(iArray.begin(), iArray.end())) * Stride) : throw(std::runtime_error("Wrong input size!"));
    }
    return (nSize > 0) ? static_cast<int>(std::distance(idata, std::max_element(idata, idata + nSize))) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline int stats::maxelement_index(std::vector<DataType> const &idata) const
{
    return (idata.size() > 0) ? static_cast<int>(std::distance(idata.begin(), std::max_element(idata.begin(), idata.end()))) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType>
inline int stats::maxelement_index(arrayWrapper<DataType> const &iArray) const
{
    return (iArray.size() > 0) ? static_cast<int>(std::distance(iArray.begin(), std::max_element(iArray.begin(), iArray.end())) * iArray.stride()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::sum(DataType const *idata, int const nSize, int const Stride) const
{
    if (nSize <= 0)
    {
        return OutputDataType{};
    }
    if (Stride > 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? static_cast<OutputDataType>(std::accumulate(iArray.begin(), iArray.end(), DataType{})) : OutputDataType{};
    }
#if unrolledIncrement == 0
    return static_cast<OutputDataType>(std::accumulate(idata, idata + nSize, DataType{}));
#else
    DataType s(0);
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            s += idata[i];
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
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
    return static_cast<OutputDataType>(s);
#endif
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::sum(std::vector<DataType> const &idata) const
{
    return (idata.size() > 0) ? (static_cast<OutputDataType>(std::accumulate(idata.begin(), idata.end(), DataType{}))) : OutputDataType{};
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::sum(arrayWrapper<DataType> const &iArray) const
{
    return (iArray.size() > 0) ? (static_cast<OutputDataType>(std::accumulate(iArray.begin(), iArray.end(), DataType{}))) : OutputDataType{};
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::sumAbs(DataType const *idata, int const nSize, int const Stride) const
{
    if (nSize <= 0)
    {
        return OutputDataType{};
    }
    if (Stride > 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? static_cast<OutputDataType>(std::accumulate(iArray.begin(), iArray.end(), DataType{}, [](DataType const &lhs, DataType const &rhs) { return lhs + std::abs(rhs); })) : OutputDataType{};
    }
#if unrolledIncrement == 0
    return static_cast<OutputDataType>(std::accumulate(idata, idata + nSize, DataType{}, [](DataType const &lhs, DataType const &rhs) { return lhs + std::abs(rhs); }));
#else
    DataType s(0);
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            s += std::abs(idata[i]);
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
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
    return static_cast<OutputDataType>(s);
#endif
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::sumAbs(std::vector<DataType> const &idata) const
{
    return (idata.size() > 0) ? (static_cast<OutputDataType>(std::accumulate(idata.begin(), idata.end(), DataType{}, [](DataType const &lhs, DataType const &rhs) { return lhs + std::abs(rhs); }))) : OutputDataType{};
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::sumAbs(arrayWrapper<DataType> const &iArray) const
{
    return (iArray.size() > 0) ? (static_cast<OutputDataType>(std::accumulate(iArray.begin(), iArray.end(), DataType{}, [](DataType const &lhs, DataType const &rhs) { return lhs + std::abs(rhs); }))) : OutputDataType{};
}

template <typename DataType, typename RealType>
inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::mean(DataType const *idata, int const nSize, int const Stride) const
{
    if (nSize <= 0)
    {
        UMUQFAIL("Wrong input size!");
    }
    if (Stride > 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        return (iArray.size() > 0) ? (static_cast<RealType>(std::accumulate(iArray.begin(), iArray.end(), DataType{})) / iArray.size()) : throw(std::runtime_error("Wrong input size!"));
    }
    return sum<DataType, RealType>(idata, nSize) / static_cast<RealType>(nSize);
}

template <typename DataType, typename RealType>
inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::mean(std::vector<DataType> const &idata) const
{
    return (idata.size() > 0) ? (static_cast<RealType>(std::accumulate(idata.begin(), idata.end(), DataType{})) / idata.size()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType, typename RealType>
inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::mean(arrayWrapper<DataType> const &iArray) const
{
    return (iArray.size() > 0) ? (static_cast<RealType>(std::accumulate(iArray.begin(), iArray.end(), DataType{})) / iArray.size()) : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::median(DataType const *idata, int const nSize, int const Stride)
{
    if (Stride > 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);

        // We do partial sorting algorithm that rearranges elements
        std::vector<OutputDataType> data(iArray.begin(), iArray.end());
        std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
        return (iArray.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
    }

    // We do partial sorting algorithm that rearranges elements
    std::vector<OutputDataType> data(idata, idata + nSize);
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (nSize > 1) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::median(std::vector<DataType> const &idata)
{
    // We do partial sorting algorithm that rearranges elements
    std::vector<OutputDataType> data(idata);

    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (idata.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::median(arrayWrapper<DataType> const &iArray)
{
    // We do partial sorting algorithm that rearranges elements
    std::vector<OutputDataType> data(iArray.begin(), iArray.end());

    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (iArray.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::medianAbs(DataType const *idata, int const nSize, int const Stride, OutputDataType &Median)
{
    arrayWrapper<DataType> iArray(idata, nSize, Stride);

    // We do partial sorting algorithm that rearranges elements
    std::vector<OutputDataType> data(iArray.begin(), iArray.end());

    // std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    Median = data[data.size() / 2];
    std::for_each(data.begin(), data.end(), [&](OutputDataType &d_i) { d_i = std::abs(d_i - Median); });

    // std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (nSize > 1) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::medianAbs(std::vector<DataType> const &idata, OutputDataType &Median)
{
    // We do partial sorting algorithm that rearranges elements
    std::vector<OutputDataType> data(idata);

    //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    Median = data[data.size() / 2];
    std::for_each(data.begin(), data.end(), [&](OutputDataType &d_i) { d_i = std::abs(d_i - Median); });

    //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (idata.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType, typename OutputDataType>
inline OutputDataType stats::medianAbs(arrayWrapper<DataType> const &iArray, OutputDataType &Median)
{
    // We do partial sorting algorithm that rearranges elements
    std::vector<OutputDataType> data(iArray.begin(), iArray.end());

    //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    Median = data[data.size() / 2];
    std::for_each(data.begin(), data.end(), [&](OutputDataType &d_i) { d_i = std::abs(d_i - Median); });

    //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return (iArray.size() > 0) ? data[data.size() / 2] : throw(std::runtime_error("Wrong input size!"));
}

template <typename DataType, typename RealType>
inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::stddev(DataType const *idata, int const nSize, int const Stride, RealType const idataMean) const
{
    RealType const Mean = idataMean < std::numeric_limits<RealType>::max() ? idataMean : mean<DataType, RealType>(idata, nSize, Stride);
    RealType s(0);
    if (Stride != 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        std::for_each(iArray.begin(), iArray.end(), [&](DataType const d) { s += (d - Mean) * (d - Mean); });
        return (iArray.size() > 1) ? std::sqrt(s / (iArray.size() - 1)) : std::sqrt(s);
    }
#if unrolledIncrement == 0
    std::for_each(idata, idata + nSize, [&](DataType const d) { s += (d - Mean) * (d - Mean); });
#else
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            s += (idata[i] - Mean) * (idata[i] - Mean);
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
    {
        RealType const diff0 = idata[i] - Mean;
        RealType const diff1 = idata[i + 1] - Mean;
        RealType const diff2 = idata[i + 2] - Mean;
        RealType const diff3 = idata[i + 3] - Mean;
#if unrolledIncrement == 4
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
#endif
#if unrolledIncrement == 6
        RealType const diff4 = idata[i + 4] - Mean;
        RealType const diff5 = idata[i + 5] - Mean;
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5;
#endif
#if unrolledIncrement == 8
        RealType const diff4 = idata[i + 4] - Mean;
        RealType const diff5 = idata[i + 5] - Mean;
        RealType const diff6 = idata[i + 6] - Mean;
        RealType const diff7 = idata[i + 7] - Mean;
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
#endif
#if unrolledIncrement == 10
        RealType const diff4 = idata[i + 4] - Mean;
        RealType const diff5 = idata[i + 5] - Mean;
        RealType const diff6 = idata[i + 6] - Mean;
        RealType const diff7 = idata[i + 7] - Mean;
        RealType const diff8 = idata[i + 8] - Mean;
        RealType const diff9 = idata[i + 9] - Mean;
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9;
#endif
#if unrolledIncrement == 12
        RealType const diff4 = idata[i + 4] - Mean;
        RealType const diff5 = idata[i + 5] - Mean;
        RealType const diff6 = idata[i + 6] - Mean;
        RealType const diff7 = idata[i + 7] - Mean;
        RealType const diff8 = idata[i + 8] - Mean;
        RealType const diff9 = idata[i + 9] - Mean;
        RealType const diff10 = idata[i + 10] - Mean;
        RealType const diff11 = idata[i + 11] - Mean;
        s += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7 + diff8 * diff8 + diff9 * diff9 + diff10 * diff10 + diff11 * diff11;
#endif
    }
#endif
    return (nSize > 1) ? std::sqrt(s / (nSize - 1)) : std::sqrt(s);
}

template <typename DataType, typename RealType>
inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::stddev(std::vector<DataType> const &idata, RealType const idataMean) const
{
    RealType const Mean = idataMean < std::numeric_limits<RealType>::max() ? idataMean : mean<DataType, RealType>(idata);
    RealType s(0);
    std::for_each(idata.begin(), idata.end(), [&](DataType const d) { s += (d - Mean) * (d - Mean); });
    return (idata.size() > 1) ? std::sqrt(s / (idata.size() - 1)) : std::sqrt(s);
}

template <typename DataType, typename RealType>
inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::stddev(arrayWrapper<DataType> const &iArray, RealType const idataMean) const
{
    RealType const Mean = idataMean < std::numeric_limits<RealType>::max() ? idataMean : mean<DataType, RealType>(iArray);
    RealType s(0);
    std::for_each(iArray.begin(), iArray.end(), [&](DataType const d) { s += (d - Mean) * (d - Mean); });
    return (iArray.size() > 1) ? std::sqrt(s / (iArray.size() - 1)) : std::sqrt(s);
}

template <typename DataType, typename RealType>
inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::coefvar(DataType const *idata, int const nSize, int const Stride, RealType const idataMean) const
{
    RealType const Mean = idataMean < std::numeric_limits<RealType>::max() ? idataMean : mean<DataType, RealType>(idata, nSize, Stride);
    RealType const s = stddev<DataType, RealType>(idata, nSize, Stride, Mean);
    return s / Mean;
}

template <typename DataType, typename RealType>
inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::coefvar(std::vector<DataType> const &idata, RealType const idataMean) const
{
    RealType const Mean = idataMean < std::numeric_limits<RealType>::max() ? idataMean : mean<DataType, RealType>(idata);
    RealType s(0);
    std::for_each(idata.begin(), idata.end(), [&](DataType const d) { s += (d - Mean) * (d - Mean); });
    return (idata.size() > 1) ? std::sqrt(s / (idata.size() - 1)) / Mean : std::sqrt(s) / Mean;
}

template <typename DataType, typename RealType>
inline std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::coefvar(arrayWrapper<DataType> const &iArray, RealType const idataMean) const
{
    RealType const Mean = idataMean < std::numeric_limits<RealType>::max() ? idataMean : mean<DataType, RealType>(iArray);
    RealType s(0);
    std::for_each(iArray.begin(), iArray.end(), [&](DataType const d) { s += (d - Mean) * (d - Mean); });
    return (iArray.size() > 1) ? std::sqrt(s / (iArray.size() - 1)) / Mean : std::sqrt(s) / Mean;
}

template <typename DataType>
void stats::minmaxNormal(DataType *idata, int const nSize, int const Stride, DataType const MinValue, DataType const MaxValue)
{
    DataType const minValue = (MinValue <= UFAIL) ? minelement<DataType>(idata, nSize, Stride) : MinValue;
    DataType const maxValue = (MaxValue <= UFAIL) ? maxelement<DataType>(idata, nSize, Stride) : MaxValue;

    DataType minmaxDenominator = maxValue - minValue;
    if (minmaxDenominator < std::numeric_limits<DataType>::epsilon())
    {
        UMUQWARNING("Maximum and Minimum Value are identical!");
        minmaxDenominator = std::numeric_limits<DataType>::epsilon();
    }
    if (Stride > 1)
    {
        for (int i = 0; i < nSize; i += Stride)
        {
            idata[i] -= minValue;
            idata[i] /= minmaxDenominator;
        }
        return;
    }
#if unrolledIncrement == 0
    std::for_each(idata, idata + nSize, [&](DataType &d_i) { d_i = (d_i - minValue) / minmaxDenominator; });
#else
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            idata[i] -= minValue;
            idata[i] /= minmaxDenominator;
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
    {
        idata[i] -= minValue;
        idata[i] /= minmaxDenominator;
        idata[i + 1] -= minValue;
        idata[i + 1] /= minmaxDenominator;
        idata[i + 2] -= minValue;
        idata[i + 2] /= minmaxDenominator;
        idata[i + 3] -= minValue;
        idata[i + 3] /= minmaxDenominator;
#if unrolledIncrement == 6
        idata[i + 4] -= minValue;
        idata[i + 4] /= minmaxDenominator;
        idata[i + 5] -= minValue;
        idata[i + 5] /= minmaxDenominator;
#endif
#if unrolledIncrement == 8
        idata[i + 4] -= minValue;
        idata[i + 4] /= minmaxDenominator;
        idata[i + 5] -= minValue;
        idata[i + 5] /= minmaxDenominator;
        idata[i + 6] -= minValue;
        idata[i + 6] /= minmaxDenominator;
        idata[i + 7] -= minValue;
        idata[i + 7] /= minmaxDenominator;
#endif
#if unrolledIncrement == 10
        idata[i + 4] -= minValue;
        idata[i + 4] /= minmaxDenominator;
        idata[i + 5] -= minValue;
        idata[i + 5] /= minmaxDenominator;
        idata[i + 6] -= minValue;
        idata[i + 6] /= minmaxDenominator;
        idata[i + 7] -= minValue;
        idata[i + 7] /= minmaxDenominator;
        idata[i + 8] -= minValue;
        idata[i + 8] /= minmaxDenominator;
        idata[i + 9] -= minValue;
        idata[i + 9] /= minmaxDenominator;
#endif
#if unrolledIncrement == 12
        idata[i + 4] -= minValue;
        idata[i + 4] /= minmaxDenominator;
        idata[i + 5] -= minValue;
        idata[i + 5] /= minmaxDenominator;
        idata[i + 6] -= minValue;
        idata[i + 6] /= minmaxDenominator;
        idata[i + 7] -= minValue;
        idata[i + 7] /= minmaxDenominator;
        idata[i + 8] -= minValue;
        idata[i + 8] /= minmaxDenominator;
        idata[i + 9] -= minValue;
        idata[i + 9] /= minmaxDenominator;
        idata[i + 10] -= minValue;
        idata[i + 10] /= minmaxDenominator;
        idata[i + 11] -= minValue;
        idata[i + 11] /= minmaxDenominator;
#endif
    }
#endif
    return;
}

template <typename DataType>
void stats::minmaxNormal(DataType *idata, int const nSize, DataType &MinValue, DataType &MaxValue, int const Stride)
{
    MinValue = minelement<DataType>(idata, nSize, Stride);
    MaxValue = maxelement<DataType>(idata, nSize, Stride);

    DataType minmaxDenominator = MaxValue - MinValue;
    if (minmaxDenominator < std::numeric_limits<DataType>::epsilon())
    {
        UMUQWARNING("Maximum and Minimum Value are identical!");
        minmaxDenominator = std::numeric_limits<DataType>::epsilon();
    }
    if (Stride > 1)
    {
        for (int i = 0; i < nSize; i += Stride)
        {
            idata[i] -= MinValue;
            idata[i] /= minmaxDenominator;
        }
        return;
    }
#if unrolledIncrement == 0
    std::for_each(idata, idata + nSize, [&](DataType &d_i) { d_i = (d_i - MinValue) / minmaxDenominator; });
#else
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            idata[i] -= MinValue;
            idata[i] /= minmaxDenominator;
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
    {
        idata[i] -= MinValue;
        idata[i] /= minmaxDenominator;
        idata[i + 1] -= MinValue;
        idata[i + 1] /= minmaxDenominator;
        idata[i + 2] -= MinValue;
        idata[i + 2] /= minmaxDenominator;
        idata[i + 3] -= MinValue;
        idata[i + 3] /= minmaxDenominator;
#if unrolledIncrement == 6
        idata[i + 4] -= MinValue;
        idata[i + 4] /= minmaxDenominator;
        idata[i + 5] -= MinValue;
        idata[i + 5] /= minmaxDenominator;
#endif
#if unrolledIncrement == 8
        idata[i + 4] -= MinValue;
        idata[i + 4] /= minmaxDenominator;
        idata[i + 5] -= MinValue;
        idata[i + 5] /= minmaxDenominator;
        idata[i + 6] -= MinValue;
        idata[i + 6] /= minmaxDenominator;
        idata[i + 7] -= MinValue;
        idata[i + 7] /= minmaxDenominator;
#endif
#if unrolledIncrement == 10
        idata[i + 4] -= MinValue;
        idata[i + 4] /= minmaxDenominator;
        idata[i + 5] -= MinValue;
        idata[i + 5] /= minmaxDenominator;
        idata[i + 6] -= MinValue;
        idata[i + 6] /= minmaxDenominator;
        idata[i + 7] -= MinValue;
        idata[i + 7] /= minmaxDenominator;
        idata[i + 8] -= MinValue;
        idata[i + 8] /= minmaxDenominator;
        idata[i + 9] -= MinValue;
        idata[i + 9] /= minmaxDenominator;
#endif
#if unrolledIncrement == 12
        idata[i + 4] -= MinValue;
        idata[i + 4] /= minmaxDenominator;
        idata[i + 5] -= MinValue;
        idata[i + 5] /= minmaxDenominator;
        idata[i + 6] -= MinValue;
        idata[i + 6] /= minmaxDenominator;
        idata[i + 7] -= MinValue;
        idata[i + 7] /= minmaxDenominator;
        idata[i + 8] -= MinValue;
        idata[i + 8] /= minmaxDenominator;
        idata[i + 9] -= MinValue;
        idata[i + 9] /= minmaxDenominator;
        idata[i + 10] -= MinValue;
        idata[i + 10] /= minmaxDenominator;
        idata[i + 11] -= MinValue;
        idata[i + 11] /= minmaxDenominator;
#endif
    }
#endif
    return;
}

template <typename DataType>
void stats::minmaxNormal(std::vector<DataType> &idata, DataType const MinValue, DataType const MaxValue)
{
    DataType const minValue = (MinValue <= UFAIL) ? minelement<DataType>(idata) : MinValue;
    DataType const maxValue = (MaxValue <= UFAIL) ? maxelement<DataType>(idata) : MaxValue;

    DataType minmaxDenominator = maxValue - minValue;
    if (minmaxDenominator < std::numeric_limits<DataType>::epsilon())
    {
        UMUQWARNING("Maximum and Minimum Value are identical!");
        minmaxDenominator = std::numeric_limits<DataType>::epsilon();
    }

    std::for_each(idata.begin(), idata.end(), [&](DataType &d_i) { d_i = (d_i - minValue) / minmaxDenominator; });
    return;
}

template <typename DataType>
void stats::minmaxNormal(std::vector<DataType> &idata, DataType &MinValue, DataType &MaxValue)
{
    MinValue = minelement<DataType>(idata);
    MaxValue = maxelement<DataType>(idata);

    DataType minmaxDenominator = MaxValue - MinValue;
    if (minmaxDenominator < std::numeric_limits<DataType>::epsilon())
    {
        UMUQWARNING("Maximum and Minimum Value are identical!");
        minmaxDenominator = std::numeric_limits<DataType>::epsilon();
    }

    std::for_each(idata.begin(), idata.end(), [&](DataType &d_i) { d_i = (d_i - MinValue) / minmaxDenominator; });
    return;
}

template <typename DataType>
inline void stats::zscoreNormal(DataType *idata, int const nSize, int const Stride)
{
    DataType const MeanValue = mean<DataType, DataType>(idata, nSize, Stride);
    DataType const StddevValue = stddev<DataType, DataType>(idata, nSize, Stride, MeanValue);
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
    std::for_each(idata, idata + nSize, [&](DataType &d_i) { d_i = (d_i - MeanValue) / StddevValue; });
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

template <typename DataType>
inline void stats::zscoreNormal(std::vector<DataType> &idata)
{
    DataType const MeanValue = mean<DataType, DataType>(idata);
    DataType const StddevValue = stddev<DataType, DataType>(idata, MeanValue);
    std::for_each(idata.begin(), idata.end(), [&](DataType &d_i) { d_i = (d_i - MeanValue) / StddevValue; });
    return;
}

template <typename DataType>
inline void stats::robustzscoreNormal(DataType *idata, int const nSize, int const Stride)
{
    DataType Median;
    DataType const mad = medianAbs<DataType, DataType>(idata, nSize, Stride, Median);
    if (Stride > 1)
    {
        for (int i = 0; i < nSize; i += Stride)
        {
            idata[i] -= Median;
            idata[i] /= mad;
        }
        return;
    }
#if unrolledIncrement == 0
    std::for_each(idata, idata + nSize, [&](DataType &d_i) { d_i = (d_i - Median) / mad; });
#else
    int const n = (nSize > unrolledIncrement) ? (nSize % unrolledIncrement) : nSize;
    if (n)
    {
        for (int i = 0; i < n; i++)
        {
            idata[i] -= Median;
            idata[i] /= mad;
        }
    }
    for (int i = n; i < nSize; i += unrolledIncrement)
    {
        idata[i] -= Median;
        idata[i] /= mad;
        idata[i + 1] -= Median;
        idata[i + 1] /= mad;
        idata[i + 2] -= Median;
        idata[i + 2] /= mad;
        idata[i + 3] -= Median;
        idata[i + 3] /= mad;
#if unrolledIncrement == 6
        idata[i + 4] -= Median;
        idata[i + 4] /= mad;
        idata[i + 5] -= Median;
        idata[i + 5] /= mad;
#endif
#if unrolledIncrement == 8
        idata[i + 4] -= Median;
        idata[i + 4] /= mad;
        idata[i + 5] -= Median;
        idata[i + 5] /= mad;
        idata[i + 6] -= Median;
        idata[i + 6] /= mad;
        idata[i + 7] -= Median;
        idata[i + 7] /= mad;
#endif
#if unrolledIncrement == 10
        idata[i + 4] -= Median;
        idata[i + 4] /= mad;
        idata[i + 5] -= Median;
        idata[i + 5] /= mad;
        idata[i + 6] -= Median;
        idata[i + 6] /= mad;
        idata[i + 7] -= Median;
        idata[i + 7] /= mad;
        idata[i + 8] -= Median;
        idata[i + 8] /= mad;
        idata[i + 9] -= Median;
        idata[i + 9] /= mad;
#endif
#if unrolledIncrement == 12
        idata[i + 4] -= Median;
        idata[i + 4] /= mad;
        idata[i + 5] -= Median;
        idata[i + 5] /= mad;
        idata[i + 6] -= Median;
        idata[i + 6] /= mad;
        idata[i + 7] -= Median;
        idata[i + 7] /= mad;
        idata[i + 8] -= Median;
        idata[i + 8] /= mad;
        idata[i + 9] -= Median;
        idata[i + 9] /= mad;
        idata[i + 10] -= Median;
        idata[i + 10] /= mad;
        idata[i + 11] -= Median;
        idata[i + 11] /= mad;
#endif
    }
#endif
    return;
}

template <typename DataType>
inline void stats::robustzscoreNormal(std::vector<DataType> &idata)
{
    DataType Median;
    DataType const mad = medianAbs<DataType, DataType>(idata, Median);
    std::for_each(idata.begin(), idata.end(), [&](DataType &d_i) { d_i = (d_i - Median) / mad; });
    return;
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::covariance(DataType const *idata, DataType const *jdata, int const nSize, RealType const iMean, RealType const jMean)
{
    /*!
     * \todo
     * If the data size is too big, maybe we should force long double
     */
    RealType Covariance(0);
    for (int i = 0; i < nSize; i++)
    {
        RealType const d1 = idata[i] - iMean;
        RealType const d2 = jdata[i] - jMean;
        Covariance += (d1 * d2 - Covariance) / static_cast<RealType>(i + 1);
    }
    return (nSize > 1) ? Covariance * static_cast<RealType>(nSize) / static_cast<RealType>(nSize - 1) : Covariance;
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::covariance(arrayWrapper<DataType> const &iArray, arrayWrapper<DataType> const &jArray, RealType const iMean, RealType const jMean)
{
    RealType Covariance(0);
    int iSize = 1;
    for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
    {
        RealType const d1 = *i - iMean;
        RealType const d2 = *j - jMean;
        Covariance += (d1 * d2 - Covariance) / static_cast<RealType>(iSize);
        iSize++;
    }
    --iSize;
    return (iSize > 1) ? Covariance * static_cast<RealType>(iSize) / static_cast<RealType>(iSize - 1) : Covariance;
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::covariance(std::vector<DataType> const &idata, std::vector<DataType> const &jdata, RealType const iMean, RealType const jMean)
{
    RealType Covariance(0);
    int iSize = 1;
    for (auto i = idata.begin(), j = jdata.begin(); i != idata.end(); i++, j++)
    {
        RealType const d1 = *i - iMean;
        RealType const d2 = *j - jMean;
        Covariance += (d1 * d2 - Covariance) / static_cast<RealType>(iSize);
        iSize++;
    }
    --iSize;
    return (iSize > 1) ? Covariance * static_cast<RealType>(iSize) / static_cast<RealType>(iSize - 1) : Covariance;
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::covariance(DataType const *idata, DataType const *jdata, int const nSize, int const Stride)
{
    RealType const iMean = mean<DataType, RealType>(idata, nSize, Stride);
    RealType const jMean = mean<DataType, RealType>(jdata, nSize, Stride);

    RealType Covariance(0);

    if (Stride != 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        arrayWrapper<DataType> jArray(jdata, nSize, Stride);

        int iSize = 1;
        for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
        {
            RealType const d1 = *i - iMean;
            RealType const d2 = *j - jMean;
            Covariance += (d1 * d2 - Covariance) / static_cast<RealType>(iSize);
            iSize++;
        }
        --iSize;
        return (iSize > 1) ? Covariance * static_cast<RealType>(iSize) / static_cast<RealType>(iSize - 1) : Covariance;
    }

    for (int i = 0; i < nSize; i++)
    {
        RealType const d1 = idata[i] - iMean;
        RealType const d2 = jdata[i] - jMean;
        Covariance += (d1 * d2 - Covariance) / static_cast<RealType>(i + 1);
    }

    return (nSize > 1) ? Covariance * static_cast<RealType>(nSize) / static_cast<RealType>(nSize - 1) : Covariance;
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::covariance(arrayWrapper<DataType> const &iArray, arrayWrapper<DataType> const &jArray)
{
    RealType const iMean = mean<DataType, RealType>(iArray);
    RealType const jMean = mean<DataType, RealType>(jArray);

    RealType Covariance(0);

    int iSize = 1;
    for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
    {
        RealType const d1 = *i - iMean;
        RealType const d2 = *j - jMean;
        Covariance += (d1 * d2 - Covariance) / static_cast<RealType>(iSize);
        iSize++;
    }
    --iSize;
    return (iSize > 1) ? Covariance * static_cast<RealType>(iSize) / static_cast<RealType>(iSize - 1) : Covariance;
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::covariance(std::vector<DataType> const &idata, std::vector<DataType> const &jdata)
{
    RealType const iMean = mean<DataType, RealType>(idata);
    RealType const jMean = mean<DataType, RealType>(jdata);

    RealType Covariance(0);

    int iSize = 1;
    for (auto i = idata.begin(), j = jdata.begin(); i != idata.end(); i++, j++)
    {
        RealType const d1 = *i - iMean;
        RealType const d2 = *j - jMean;
        Covariance += (d1 * d2 - Covariance) / static_cast<RealType>(iSize);
        iSize++;
    }
    --iSize;
    return (iSize > 1) ? Covariance * static_cast<RealType>(iSize) / static_cast<RealType>(iSize - 1) : Covariance;
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    *stats::covariance(DataType const *idata, int const nSize, int const nDim, int const Stride)
{
    RealType *Covariance = nullptr;
    try
    {
        Covariance = new RealType[nDim * nDim]();
    }
    catch (std::bad_alloc &e)
    {
        UMUQFAILRETURNNULL("Failed to allocate memory!");
    }

    std::vector<RealType> iMean(nDim);

    // We should make sure of the correct stride
    int const stride = Stride > nDim ? Stride : nDim;

    // Compute the mean for each dimension
    for (int i = 0; i < nDim; i++)
    {
        iMean[i] = mean<DataType, RealType>(idata + i, nSize, stride);
    }

    for (int i = 0; i < nDim; i++)
    {
        arrayWrapper<DataType> iArray(idata + i, nSize, stride);

        for (int j = i; j < nDim; j++)
        {
            arrayWrapper<DataType> jArray(idata + j, nSize, stride);

            Covariance[i * nDim + j] = covariance<DataType, RealType>(iArray, jArray, iMean[i], iMean[j]);
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

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    *stats::covariance(DataType const *idata, int const nSize, int const nDim, RealType const *iMean)
{
    RealType *Covariance = nullptr;
    try
    {
        Covariance = new RealType[nDim * nDim]();
    }
    catch (std::bad_alloc &e)
    {
        UMUQFAILRETURNNULL("Failed to allocate memory!");
    }

    for (int i = 0; i < nDim; i++)
    {
        arrayWrapper<DataType> iArray(idata + i, nSize, nDim);

        for (int j = i; j < nDim; j++)
        {
            arrayWrapper<DataType> jArray(idata + j, nSize, nDim);

            Covariance[i * nDim + j] = covariance<DataType, RealType>(iArray, jArray, iMean[i], iMean[j]);
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

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::correlation(DataType const *idata, DataType const *jdata, int const nSize, RealType const iMean, RealType const jMean)
{
    RealType const iStddev = stddev<DataType, RealType>(idata, nSize, 1, iMean);
    RealType const jStddev = stddev<DataType, RealType>(jdata, nSize, 1, jMean);
    RealType const Covariance = covariance<DataType, RealType>(idata, jdata, nSize, iMean, jMean);
    return Covariance / (iStddev * jStddev);
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::correlation(arrayWrapper<DataType> const &iArray, arrayWrapper<DataType> const &jArray, RealType const iMean, RealType const jMean)
{
    RealType const iStddev = stddev<DataType, RealType>(iArray, iMean);
    RealType const jStddev = stddev<DataType, RealType>(jArray, jMean);
    RealType const Covariance = covariance<DataType, RealType>(iArray, jArray, iMean, jMean);
    return Covariance / (iStddev * jStddev);
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::correlation(std::vector<DataType> const &idata, std::vector<DataType> const &jdata, RealType const iMean, RealType const jMean)
{
    RealType const iStddev = stddev<DataType, RealType>(idata, iMean);
    RealType const jStddev = stddev<DataType, RealType>(jdata, jMean);
    RealType const Covariance = covariance<DataType, RealType>(idata, jdata, iMean, jMean);
    return Covariance / (iStddev * jStddev);
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::correlation(DataType const *idata, DataType const *jdata, int const nSize, int const Stride)
{
    if (Stride != 1)
    {
        arrayWrapper<DataType> iArray(idata, nSize, Stride);
        if (iArray.size())
        {
            RealType idataSumSquare(0);
            RealType jdataSumSquare(0);
            RealType idatajdataCross(0);

            arrayWrapper<DataType> jArray(jdata, nSize, Stride);

            RealType iMean = iArray[0];
            RealType jMean = jArray[0];

            int iSize = 1;
            for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
            {
                RealType const ratio = iSize / (iSize + 1.0);
                RealType const d1 = *i - iMean;
                RealType const d2 = *j - jMean;
                idataSumSquare += d1 * d1 * ratio;
                jdataSumSquare += d2 * d2 * ratio;
                idatajdataCross += d1 * d2 * ratio;
                iMean += d1 / (iSize + 1.0);
                jMean += d2 / (iSize + 1.0);
                iSize++;
            }
            return idatajdataCross / (std::sqrt(idataSumSquare) * std::sqrt(jdataSumSquare));
        }
        UMUQFAIL("The input idata with the requested stride ", Stride, " is empty!");
    }
    else
    {
        RealType idataSumSquare(0);
        RealType jdataSumSquare(0);
        RealType idatajdataCross(0);
        RealType iMean = idata[0];
        RealType jMean = jdata[0];

        for (int i = 1; i < nSize; i++)
        {
            RealType const ratio = i / (i + 1.0);
            RealType const d1 = idata[i] - iMean;
            RealType const d2 = jdata[i] - jMean;
            idataSumSquare += d1 * d1 * ratio;
            jdataSumSquare += d2 * d2 * ratio;
            idatajdataCross += d1 * d2 * ratio;
            iMean += d1 / (i + 1.0);
            jMean += d2 / (i + 1.0);
        }
        return idatajdataCross / (std::sqrt(idataSumSquare) * std::sqrt(jdataSumSquare));
    }
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::correlation(arrayWrapper<DataType> const &iArray, arrayWrapper<DataType> const &jArray)
{
    if (iArray.size())
    {
        if (iArray.size() != jArray.size())
        {
            UMUQFAIL("Input arrays have different sizes!");
        }

        RealType idataSumSquare(0);
        RealType jdataSumSquare(0);
        RealType idatajdataCross(0);
        RealType iMean = iArray[0];
        RealType jMean = jArray[0];

        int iSize = 1;
        for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
        {
            RealType const ratio = iSize / (iSize + 1.0);
            RealType const d1 = *i - iMean;
            RealType const d2 = *j - jMean;
            idataSumSquare += d1 * d1 * ratio;
            jdataSumSquare += d2 * d2 * ratio;
            idatajdataCross += d1 * d2 * ratio;
            iMean += d1 / (iSize + 1.0);
            jMean += d2 / (iSize + 1.0);
            iSize++;
        }
        return idatajdataCross / (std::sqrt(idataSumSquare) * std::sqrt(jdataSumSquare));
    }
    UMUQFAIL("The input array is empty!");
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
stats::correlation(std::vector<DataType> const &idata, std::vector<DataType> const &jdata)
{
    if (idata.size())
    {
        if (idata.size() != jdata.size())
        {
            UMUQFAIL("Input vectors have different sizes!");
        }

        RealType idataSumSquare(0);
        RealType jdataSumSquare(0);
        RealType idatajdataCross(0);
        RealType iMean = idata[0];
        RealType jMean = jdata[0];

        int iSize = 1;
        for (auto i = idata.begin(), j = jdata.begin(); i != idata.end(); i++, j++)
        {
            RealType const ratio = iSize / (iSize + 1.0);
            RealType const d1 = *i - iMean;
            RealType const d2 = *j - jMean;
            idataSumSquare += d1 * d1 * ratio;
            jdataSumSquare += d2 * d2 * ratio;
            idatajdataCross += d1 * d2 * ratio;
            iMean += d1 / (iSize + 1.0);
            jMean += d2 / (iSize + 1.0);
            iSize++;
        }
        return idatajdataCross / (std::sqrt(idataSumSquare) * std::sqrt(jdataSumSquare));
    }
    UMUQFAIL("The input vector is empty!");
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    *stats::correlation(DataType const *idata, int const nSize, int const nDim, int const Stride)
{
    RealType *Covariance = covariance<DataType, RealType>(idata, nSize, nDim, Stride);
    if (Covariance)
    {
        EMapType<RealType> CovarianceMatrix(Covariance, nDim, nDim);
        EVectorX<RealType> CovarianceMatrixDiagonal = CovarianceMatrix.diagonal().cwiseSqrt().cwiseInverse();
        CovarianceMatrix = CovarianceMatrixDiagonal.asDiagonal() * CovarianceMatrix * CovarianceMatrixDiagonal.asDiagonal();
        return Covariance;
    }
    UMUQFAILRETURNNULL("Failed to compute the covariance!");
}

template <typename DataType, typename RealType>
std::enable_if_t<std::is_floating_point<RealType>::value, RealType>
    *stats::correlation(DataType const *idata, int const nSize, int const nDim, RealType const *iMean)
{
    RealType *Covariance = covariance<DataType, RealType>(idata, nSize, nDim, iMean);
    if (Covariance)
    {
        EMapType<RealType> CovarianceMatrix(Covariance, nDim, nDim);
        EVectorX<RealType> CovarianceMatrixDiagonal = CovarianceMatrix.diagonal().cwiseSqrt().cwiseInverse();
        CovarianceMatrix = CovarianceMatrixDiagonal.asDiagonal() * CovarianceMatrix * CovarianceMatrixDiagonal.asDiagonal();
        return Covariance;
    }
    UMUQFAILRETURNNULL("Failed to compute the covariance!");
}

template <typename DataType>
void stats::unique(DataType const *idata, int const nRows, int const nCols, std::vector<DataType> &udata)
{
    if (udata.size() < nRows * nCols)
    {
        // Resize the unique array to the maximum size
        udata.resize(nRows * nCols);
    }

    // Create a temporary array with the size of number of columns (one row of data)
    std::vector<DataType> x(nCols);

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

template <typename DataType>
void stats::unique(std::vector<DataType> const &idata, int const nRows, int const nCols, std::vector<DataType> &udata)
{
    unique(idata.data(), nRows, nCols, udata);
}

template <typename DataType>
void stats::unique(std::unique_ptr<DataType[]> const &idata, int const nRows, int const nCols, std::vector<DataType> &udata)
{
    unique(idata.get(), nRows, nCols, udata);
}

} // namespace umuq

#endif // UMUQ_STATS
