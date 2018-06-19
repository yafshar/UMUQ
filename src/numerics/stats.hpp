#ifndef UMUQ_STATS_H
#define UMUQ_STATS_H

#include "../misc/array.hpp"

/*! \class stats
* \brief stats is a class which includes some functionality for statistics of the input data
*
* It includes:
* \b minelement
* \b maxelement
* \b minelement_index
* \b maxelement_index
* \b sum
* \b mean
* \b median
* \b medianAbs
* \b stddev
* \b CoefVar
* \b minmaxNormal
* \b zscoreNormal
* \b robustzscoreNormal
*/
struct stats
{
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
            ArrayWrapper<T> a(idata, nSize, Stride);
            return *std::min_element(a.begin(), a.end());
        }
        return *std::min_element(idata, idata + nSize);
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
            ArrayWrapper<T> a(idata, nSize, Stride);
            return *std::max_element(a.begin(), a.end());
        }
        return *std::max_element(idata, idata + nSize);
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
            ArrayWrapper<T> a(idata, nSize, Stride);
            return static_cast<int>(std::distance(a.begin(), std::min_element(a.begin(), a.end())) * Stride);
        }
        return static_cast<int>(std::distance(idata, std::min_element(idata, idata + nSize)));
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
            ArrayWrapper<T> a(idata, nSize, Stride);
            return static_cast<int>(std::distance(a.begin(), std::max_element(a.begin(), a.end())) * Stride);
        }
        return static_cast<int>(std::distance(idata, std::max_element(idata, idata + nSize)));
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
            ArrayWrapper<T> a(idata, nSize, Stride);
            return static_cast<TOut>(std::accumulate(a.begin(), a.end(), T{}));
        }
        return static_cast<TOut>(std::accumulate(idata, idata + nSize, T{}));
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
            ArrayWrapper<T> a(idata, nSize, Stride);
            return static_cast<TOut>(std::accumulate(a.begin(), a.end(), T{})) / a.size();
        }
        return static_cast<TOut>(std::accumulate(idata, idata + nSize, T{})) / nSize;
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
        if (Stride != 1)
        {
            ArrayWrapper<T> a(idata, nSize, Stride);
            std::vector<TOut> data(a.begin(), a.end());
            std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
            return data[data.size() / 2];
        }
        std::vector<TOut> data(idata, idata + nSize);
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
        ArrayWrapper<T> a(idata, nSize, Stride);
        std::vector<TOut> data(a.begin(), a.end());

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
        TOut m = (idatamean < std::numeric_limits<TOut>::max() ? idatamean : mean<T, TOut>(idata, nSize, Stride));
        TOut s(0);
        if (Stride != 1)
        {
            ArrayWrapper<T> a(idata, nSize, Stride);
            std::for_each(a.begin(), a.end(), [&](T const d) { s += (d - m) * (d - m); });
            return a.size() > 1 ? std::sqrt(s / (a.size() - 1)) : std::sqrt(s);
        }
        std::for_each(idata, idata + nSize, [&](T const d) { s += (d - m) * (d - m); });
        return nSize > 1 ? std::sqrt(s / (nSize - 1)) : std::sqrt(s);
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
        ArrayWrapper<T> a(fValue, fSize, Stride);
        auto start = a.begin();
        auto end = a.end();
        int aSize = a.size();

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
            ArrayWrapper<T> a(idata, nSize, Stride);
            std::for_each(a.begin(), a.end(), [&](T &d_i) { d_i = (d_i - MinValue) / denom; });
            return;
        }
        std::for_each(idata, idata + nSize, [&](T &d_i) { d_i = (d_i - MinValue) / denom; });
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
};

#endif
