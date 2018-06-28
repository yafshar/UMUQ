#ifndef UMUQ_STATS_H
#define UMUQ_STATS_H

#include "../misc/array.hpp"

/*! \class stats
* \brief stats is a class which includes some functionality for statistics of the input data
*
* It includes:
* \b minelement         Finds the smallest element in the array of data
* \b maxelement         Finds the greatest element in the array of data
* \b minelement_index   Finds the position of the smallest element in the array of data 
* \b maxelement_index   Finds the position of the greatest element in the array of data 
* \b sum                Computes the sum of the elements in the array of data
* \b mean               Computes the mean of the elements in the array of data
* \b median             Computes the mdeian of the elements in the array of data
* \b medianAbs          Computes the median absolute deviation (MAD) of the elements in the array of data
* \b stddev             Computes the standard deviation of the elements in the array of data
* \b CoefVar            Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold
* \b minmaxNormal       Scales the numeric data using the MinMax normalization method
* \b zscoreNormal       Scales the numeric data using the Z-score normalization method
* \b robustzscoreNormal Scales the numeric data using the robust Z-score normalization method
* \b covariance         Compute the covariance
*/
struct stats
{
    /*!
     * \brief Construct a new stats object
     * 
     */
    stats() {}

    /*!
     * \brief Destroy the stats object
     * 
     */
    ~stats() {}

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
    inline T minelement(T const *idata, int const nSize, std::size_t const Stride = 1) const
    {
        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);
            return *std::min_element(iArray.begin(), iArray.end());
        }
        return *std::min_element(idata, idata + nSize);
    }

    template <typename T>
    inline T minelement(ArrayWrapper<T> const &iArray) const
    {
        return *std::min_element(iArray.begin(), iArray.end());
    }

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
    inline T maxelement(T const *idata, int const nSize, std::size_t const Stride = 1) const
    {
        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);
            return *std::max_element(iArray.begin(), iArray.end());
        }
        return *std::max_element(idata, idata + nSize);
    }

    template <typename T>
    inline T maxelement(ArrayWrapper<T> const &iArray) const
    {
        return *std::max_element(iArray.begin(), iArray.end());
    }

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
    inline int minelement_index(T const *idata, int const nSize, std::size_t const Stride = 1) const
    {
        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);
            return static_cast<int>(std::distance(iArray.begin(), std::min_element(iArray.begin(), iArray.end())) * Stride);
        }
        return static_cast<int>(std::distance(idata, std::min_element(idata, idata + nSize)));
    }

    template <typename T>
    inline int minelement_index(ArrayWrapper<T> const &iArray) const
    {
        return static_cast<int>(std::distance(iArray.begin(), std::min_element(iArray.begin(), iArray.end())) * iArray.stride());
    }

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
    inline int maxelement_index(T const *idata, int const nSize, std::size_t const Stride = 1) const
    {
        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);
            return static_cast<int>(std::distance(iArray.begin(), std::max_element(iArray.begin(), iArray.end())) * Stride);
        }
        return static_cast<int>(std::distance(idata, std::max_element(idata, idata + nSize)));
    }

    template <typename T>
    inline int maxelement_index(ArrayWrapper<T> const &iArray) const
    {
        return static_cast<int>(std::distance(iArray.begin(), std::max_element(iArray.begin(), iArray.end())) * iArray.stride());
    }

    /*!
     * \brief Computes the sum of the elements in the array of data with stride
     * 
     * \tparam T    data type
     * \tparam TOut data type of return output result (default is double)
     * 
     * \param idata array of data
     * \param nSize size of the array
     * \param Stride element stride (default is 1)
     * 
     * \returns The sum of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut sum(T const *idata, int const nSize, std::size_t const Stride = 1) const
    {
        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);
            return static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{}));
        }
        return static_cast<TOut>(std::accumulate(idata, idata + nSize, T{}));
    }

    template <typename T, typename TOut = double>
    inline TOut sum(ArrayWrapper<T> const &iArray) const
    {
        return static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{}));
    }

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
    inline TOut mean(T const *idata, const int nSize, std::size_t const Stride = 1) const
    {
        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);
            return static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{})) / iArray.size();
        }
        return static_cast<TOut>(std::accumulate(idata, idata + nSize, T{})) / nSize;
    }

    template <typename T, typename TOut = double>
    inline TOut mean(ArrayWrapper<T> const &iArray) const
    {
        return static_cast<TOut>(std::accumulate(iArray.begin(), iArray.end(), T{})) / iArray.size();
    }

    /*!
     * \brief Computes the mdeian of the elements in the array of data with Stride
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
    inline TOut median(T const *idata, const int nSize, std::size_t const Stride = 1)
    {
        if (nSize < 1)
        {
            return TOut{};
        }

        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);

            //!We do partial sorting algorithm that rearranges elements
            std::vector<TOut> data(iArray.begin(), iArray.end());
            std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
            return data[data.size() / 2];
        }

        //!We do partial sorting algorithm that rearranges elements
        std::vector<TOut> data(idata, idata + nSize);

        std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
        return data[data.size() / 2];
    }

    template <typename T, typename TOut = double>
    inline TOut median(ArrayWrapper<T> const &iArray)
    {
        if (iArray.size() < 1)
        {
            return TOut{};
        }

        //!We do partial sorting algorithm that rearranges elements
        std::vector<TOut> data(iArray.begin(), iArray.end());

        std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
        return data[data.size() / 2];
    }

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
    inline TOut medianAbs(T const *idata, const int nSize, std::size_t const Stride = 1, TOut &median_ = TOut{})
    {
        if (nSize < 1)
        {
            return TOut{};
        }

        ArrayWrapper<T> iArray(idata, nSize, Stride);

        //!We do partial sorting algorithm that rearranges elements
        std::vector<TOut> data(iArray.begin(), iArray.end());

        //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
        std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
        median_ = data[data.size() / 2];
        std::for_each(data.begin(), data.end(), [&](TOut &d_i) { d_i = std::abs(d_i - median_); });

        //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
        std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
        return data[data.size() / 2];
    }

    template <typename T, typename TOut = double>
    inline TOut medianAbs(ArrayWrapper<T> const &iArray, TOut &median_ = TOut{})
    {
        if (iArray.size() < 1)
        {
            return TOut{};
        }

        //!We do partial sorting algorithm that rearranges elements
        std::vector<TOut> data(iArray.begin(), iArray.end());

        //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
        std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
        median_ = data[data.size() / 2];
        std::for_each(data.begin(), data.end(), [&](TOut &d_i) { d_i = std::abs(d_i - median_); });

        //std::nth_element partial sorting algorithm that rearranges elements in [first, last)
        std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
        return data[data.size() / 2];
    }

    /*!
     * \brief Computes the standard deviation of the elements in the array of data with or without stride
     * 
     * \tparam T    data type
     * \tparam TOut data type of return output result (default is double)
     * 
     * \param idata     array of data
     * \param nSize     size of the array
     * \param Stride    element stride (optional, default is 1)
     * \param idatamean mean of the elements in idata (optional)
     * 
     * \returns The standard deviation of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut stddev(T const *idata, int const nSize, std::size_t const Stride = 1, TOut const idatamean = std::numeric_limits<TOut>::max()) const
    {
        TOut m = idatamean < std::numeric_limits<TOut>::max() ? idatamean : mean<T, TOut>(idata, nSize, Stride);
        TOut s(0);
        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);
            std::for_each(iArray.begin(), iArray.end(), [&](T const d) { s += (d - m) * (d - m); });
            return iArray.size() > 1 ? std::sqrt(s / (iArray.size() - 1)) : std::sqrt(s);
        }
        std::for_each(idata, idata + nSize, [&](T const d) { s += (d - m) * (d - m); });
        return nSize > 1 ? std::sqrt(s / (nSize - 1)) : std::sqrt(s);
    }

    template <typename T, typename TOut = double>
    inline TOut stddev(ArrayWrapper<T> const &iArray, TOut const idatamean = std::numeric_limits<TOut>::max()) const
    {
        TOut m = idatamean < std::numeric_limits<TOut>::max() ? idatamean : mean<T, TOut>(iArray);
        TOut s(0);
        std::for_each(iArray.begin(), iArray.end(), [&](T const d) { s += (d - m) * (d - m); });
        return iArray.size() > 1 ? std::sqrt(s / (iArray.size() - 1)) : std::sqrt(s);
    }

    /*!
     * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold
     * 
     * \tparam T       data type
     * \tparam TOut    data type of the return output result (default is double)
     * 
     * \param  fValue  An array of log value
     * \param  fSize   size of the fValue array 
     * \param  x       
     * \param  p 
     * \param  tol     a prescribed tolerance
     * 
     * \returns the square of the coefficient of variation (COV)
     */
    template <typename T, typename TOut = double>
    TOut CoefVar(T const *fValue, int const fSize, T const x, T const p, T const tol)
    {
        //Find the maximum value in the array of fValue of size fSize
        T fMaxValue = maxelement<T>(fValue, fSize);

        TOut *weight = new TOut[fSize];

        TOut diff = static_cast<TOut>(x - p);

        //Compute the weight
        for (int i = 0; i < fSize; i++)
        {
            weight[i] = std::exp((fValue[i] - fMaxValue) * diff);
        }

        //Compute the summation of weight
        TOut weightsum = sum<TOut, TOut>(weight, fSize);

        //Normalize the weight
        std::for_each(weight, weight + fSize, [&](T &w) { w /= weightsum; });

        //Compute the mean
        TOut weightmean = mean<TOut, TOut>(weight, fSize);

        //Compute the standard deviation
        TOut weightstddev = stddev<TOut, TOut>(weight, fSize, 1, weightmean);

        delete[] weight;

        //return the square of the coefficient of variation (COV)
        return std::pow(weightstddev / weightmean - tol, 2);
    }

    /*!
     * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold
     * 
     * \tparam T       data type
     * \tparam TOut    data type of the return output result (default is double)
     * 
     * \param  fValue  An array of log value
     * \param  fSize   size of the fValue array 
     * \param  Stride  element stride 
     * \param  x       
     * \param  p 
     * \param  tol     a prescribed tolerance
     * 
     * \returns the square of the coefficient of variation (COV)
     */
    template <typename T, typename TOut = double>
    TOut CoefVar(T const *fValue, int const fSize, std::size_t const Stride, T const x, T const p, T const tol)
    {
        ArrayWrapper<T> iArray(fValue, fSize, Stride);
        auto start = iArray.begin();
        auto end = iArray.end();
        int aSize = iArray.size();

        //Find the maximum value in the array of fValue of size fSize
        T fMaxValue = *std::max_element(start, end);

        TOut *weight = new TOut[aSize];

        TOut diff = static_cast<TOut>(x - p);

        {
            int i(0);
            //Compute the weight
            for (auto it = start; it != end; it++)
            {
                weight[i++] = std::exp((*it - fMaxValue) * diff);
            }
        }

        //Compute the summation of weight
        TOut weightsum = sum<TOut, TOut>(weight, aSize);

        //Normalize the weight
        std::for_each(weight, weight + aSize, [&](T &w) { w /= weightsum; });

        //Compute the mean
        TOut weightmean = mean<TOut, TOut>(weight, aSize);

        //Compute the standard deviation
        TOut weightstddev = stddev<TOut, TOut>(weight, aSize, 1, weightmean);

        delete[] weight;

        //return the square of the coefficient of variation (COV)
        return std::pow(weightstddev / weightmean - tol, 2);
    }

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
    void minmaxNormal(T *idata, int const nSize, std::size_t const Stride = 1)
    {
        T MinValue = minelement<T>(idata, nSize, Stride);
        T MaxValue = maxelement<T>(idata, nSize, Stride);

        T denom = MaxValue - MinValue;
        if (denom < std::numeric_limits<T>::epsilon())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Maximum and Minimum Value are identical!" << std::endl;
            denom = std::numeric_limits<T>::epsilon();
        }

        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);
            std::for_each(iArray.begin(), iArray.end(), [&](T &d_i) { d_i = (d_i - MinValue) / denom; });
            return;
        }
        std::for_each(idata, idata + nSize, [&](T &d_i) { d_i = (d_i - MinValue) / denom; });
    }

    template <typename T>
    void minmaxNormal(ArrayWrapper<T> const &iArray)
    {
        T MinValue = minelement<T>(iArray);
        T MaxValue = maxelement<T>(iArray);

        T denom = MaxValue - MinValue;
        if (denom < std::numeric_limits<T>::epsilon())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Maximum and Minimum Value are identical!" << std::endl;
            denom = std::numeric_limits<T>::epsilon();
        }

        std::for_each(iArray.begin(), iArray.end(), [&](T &d_i) { d_i = (d_i - MinValue) / denom; });
        return;
    }

    /*!
     * \brief zscoreNormal scales the numeric data using the Z-score normalization method
     * 
     * Using the Z-score normalization method, one can normalize the values to be the number of 
     * standard deviations an observation is from the mean of each dimension. 
     * This allows to compare data to a normally distributed random variable.
     * 
     * \tparam T data type
     * 
     * \param idata array of data
     * \param nSize size of array
     * \param Stride element stride (default is 1)
     */
    template <typename T>
    inline void zscoreNormal(T *idata, int const nSize, std::size_t const Stride = 1)
    {
        T MeanValue = mean<T, T>(idata, nSize, Stride);
        T StddevValue = stddev<T, T>(idata, nSize, Stride, MeanValue);
        if (Stride != 1)
        {
            ArrayWrapper<T> a(idata, nSize, Stride);
            std::for_each(a.begin(), a.end(), [&](T &d_i) { d_i = (d_i - MeanValue) / StddevValue; });
            return;
        }
        std::for_each(idata, idata + nSize, [&](T &d_i) { d_i = (d_i - MeanValue) / StddevValue; });
    }

    template <typename T>
    inline void zscoreNormal(ArrayWrapper<T> const &iArray)
    {
        T MeanValue = mean<T, T>(iArray);
        T StddevValue = stddev<T, T>(iArray, MeanValue);
        std::for_each(iArray.begin(), iArray.end(), [&](T &d_i) { d_i = (d_i - MeanValue) / StddevValue; });
        return;
    }

    /*!
     * \brief robustzscoreNormal scales the numeric data using the robust Z-score normalization method
     * 
     * Using the robust Z-score normalization method, one can lessen the influence of outliers 
     * on Z-score calculations. Robust Z-score normalization uses the median value as opposed 
     * to the mean value used in Z-score. 
     * By using the median instead of the mean, it helps remove some of the influence of outliers 
     * in the data.
     * medianAbs
     * \tparam T    data type
     * 
     * \param idata array of data
     * \param nSize size of array
     * \param Stride element stride (default is 1)
     */
    template <typename T>
    inline void robustzscoreNormal(T *idata, int const nSize, std::size_t const Stride = 1)
    {
        T median_;
        T mad = medianAbs<T, T>(idata, nSize, Stride, median_);
        if (Stride != 1)
        {
            ArrayWrapper<T> a(idata, nSize, Stride);
            std::for_each(a.begin(), a.end(), [&](T &d_i) { d_i = std::abs(d_i - median_) / mad; });
            return;
        }
        std::for_each(idata, idata + nSize, [&](T &d_i) { d_i = std::abs(d_i - median_) / mad; });
    }

    template <typename T>
    inline void robustzscoreNormal(ArrayWrapper<T> const &iArray)
    {
        T median_;
        T mad = medianAbs<T, T>(iArray, median_);
        std::for_each(iArray.begin(), iArray.end(), [&](T &d_i) { d_i = std::abs(d_i - median_) / mad; });
        return;
    }

    /*!
     * \brief Compute the covariance between idata and jdata vectors which must both be of the same length nSize
     * \f$ covariance(idata, jdata) = \frac{1}{n-1} \sum_{i=1}^n (idata_i-imean)(jdata_i-jmean) \f]
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param idata  Array of data 
     * \param jdata  Array of data
     * \param nSize  Size of array
     * \param imean  Mean of idata array
     * \param jmean  Mean of jdata array
     * 
     * \returns Covariance (scaler value) between idata and jdata vectors     
     */
    template <typename T, typename TOut = double>
    TOut covariance(T const *idata, T const *jdata, int const nSize, T const imean, T const jmean)
    {
        //TODO If the data size is too big, maybe we should force long double
        TOut Covariance(0);
        for (int i = 0; i < nSize; i++)
        {
            TOut const d1 = idata[i] - imean;
            TOut const d2 = jdata[i] - jmean;
            Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(i + 1);
        }

        return nSize > 1 ? Covariance * static_cast<TOut>(nSize) / static_cast<TOut>(nSize - 1) : Covariance;
    }

    /*!
     * \brief Compute the covariance between two arrays of data which must both be of the same length
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param iArray  Array of data 
     * \param jArray  Array of data
     * \param imean   Mean of iArray 
     * \param jmean   Mean of jArray
     * 
     * \returns Covariance (scaler value) between idata and jdata vectors     
     */
    template <typename T, typename TOut = double>
    TOut covariance(ArrayWrapper<T> const &iArray, ArrayWrapper<T> const &jArray, T const imean, T const jmean)
    {
        TOut Covariance(0);
        int iSize = 1;
        for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
        {
            TOut const d1 = *i - imean;
            TOut const d2 = *j - jmean;

            Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(iSize++);
        }

        --iSize;

        return iSize > 1 ? Covariance * static_cast<TOut>(iSize) / static_cast<TOut>(iSize - 1) : Covariance;
    }

    /*!
     * \brief Compute the covariance between idata and jdata vectors which must both be of the same length nSize
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param idata  Array of data 
     * \param jdata  Array of data
     * \param nSize  Size of array
     * \param Stride sride of the data in the array (default is 1)
     * 
     * \returns Covariance (scaler value) between idata and jdata vectors     
     */
    template <typename T, typename TOut = double>
    TOut covariance(T const *idata, T const *jdata, int const nSize, std::size_t const Stride = 1)
    {
        T imean = mean<T, T>(idata, nSize, Stride);
        T jmean = mean<T, T>(jdata, nSize, Stride);

        TOut Covariance(0);
        if (Stride != 1)
        {
            ArrayWrapper<T> iArray(idata, nSize, Stride);
            ArrayWrapper<T> jArray(jdata, nSize, Stride);

            int iSize = 1;
            for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
            {
                TOut const d1 = *i - imean;
                TOut const d2 = *j - jmean;

                Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(iSize++);
            }

            --iSize;

            return iSize > 1 ? Covariance * static_cast<TOut>(iSize) / static_cast<TOut>(iSize - 1) : Covariance;
        }

        for (int i = 0; i < nSize; i++)
        {
            TOut const d1 = idata[i] - imean;
            TOut const d2 = jdata[i] - jmean;

            Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(i + 1);
        }

        return nSize > 1 ? Covariance * static_cast<TOut>(nSize) / static_cast<TOut>(nSize - 1) : Covariance;
    }

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
    TOut covariance(ArrayWrapper<T> const &iArray, ArrayWrapper<T> const &jArray)
    {
        T const imean = mean<T, T>(iArray);
        T const jmean = mean<T, T>(jArray);

        TOut Covariance(0);
        int iSize = 1;
        for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++)
        {
            TOut const d1 = *i - imean;
            TOut const d2 = *j - jmean;

            Covariance += (d1 * d2 - Covariance) / static_cast<TOut>(iSize++);
        }

        --iSize;

        return iSize > 1 ? Covariance * static_cast<TOut>(iSize) / static_cast<TOut>(iSize - 1) : Covariance;
    }

    /*!
     * \brief Compute the covariance array of N-dimensional idata
     * 
     * \tparam T     Data type (should be double or long double) 
     * \tparam TOut  Data type of the return output result (default is double)
     * 
     * \param idata  Array of N-dimensional data 
     * \param nSize  Total size of the array
     * \param nDim   Data dimension
     * \param Stride Sride of the data in the array (default is 1). 
     * 
     * The reason for having parameter stride is the case where we have coordinates and function value 
     * and would like to avoid unnecessary copying the data 
     * 
     * \returns Covariance (array of N by N) from N-dimensional idata
     */
    template <typename T, typename TOut = double>
    TOut *covariance(T const *idata, int const nSize, int const nDim, std::size_t const Stride = 1)
    {
        TOut *Covariance;
		try
		{
			Covariance = new TOut[nDim * nDim]();
		}
		catch (std::bad_alloc &e)
		{
			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
			return nullptr;
		}

        std::vector<T> imean(nDim);

        //We should make sure of the correct stride
        std::size_t const stride = Stride > static_cast<std::size_t>(nDim) ? Stride : static_cast<std::size_t>(nDim);

        //Compute the mean for each dimension
        for (int i = 0; i < nDim; i++)
        {
            imean[i] = mean<T, T>(idata + i, nSize, stride);
        }

        for (int i = 0; i < nDim; i++)
        {
            ArrayWrapper<T> iArray(idata + i, nSize, stride);

            for (int j = i; j < nDim; j++)
            {
                ArrayWrapper<T> jArray(idata + j, nSize, stride);

                Covariance[i * nDim + j] = covariance<T, TOut>(iArray, jArray, imean[i], imean[j]);
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
};

#endif
