#ifndef UMUQ_STATS_H
#define UMUQ_STATS_H

/*! \class stats
* \brief stats is a class which includes some statictics functionality
*	
*/
struct stats
{
    /*!
     * \brief Finds the smallest element in the array of data
     * 
     * \tparam T data type
     * \param idata array of data
     * \param nSize size of the array
     * 
     * \returns The smallest element in the array of data
     */
    template <typename T>
    inline T const minelement(T const *idata, int const nSize) const
    {
        return *std::min_element(idata, idata + nSize);
    }

    /*!
     * \brief Finds the greatest element in the array of data
     * 
     * \tparam T data type
     * \param idata array of data
     * \param nSize size of the array
     * 
     * \returns The greatest element in the array of data
     */
    template <typename T>
    inline T const maxelement(T const *idata, int const nSize) const
    {
        return *std::max_element(idata, idata + nSize);
    }

    /*!
     * \brief Finds the position of the smallest element in the array of data
     * 
     * \tparam T data type
     * \param idata array of data
     * \param nSize size of the array
     * 
     * 
     * \returns The the position of the smallest element
     */
    template <typename T>
    inline int const minelement_index(T const *idata, int const nSize) const
    {
        return (int)std::distance(idata, std::min_element(idata, idata + nSize));
    }

    /*!
     * \brief Finds the position of the greatest element in the array of data
     * 
     * \tparam T data type
     * \param idata array of data
     * \param nSize size of the array
     * 
     * \returns The the position of the greatest element
     */
    template <typename T>
    inline int const maxelement_index(T const *idata, int const nSize) const
    {
        return (int)std::distance(idata, std::max_element(idata, idata + nSize));
    }

    /*!
     * \brief Computes the sum of the elements in the array of data
     * 
     * \tparam T data type
     * \tparam TOut type of return output result (default is double)
     * \param idata array of data
     * \param nSize size of the array
     * 
     * \returns The sum of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut sum(T const *idata, int const nSize) const
    {
        return (TOut)std::accumulate(idata, idata + nSize, T{});
    }

    /*!
     * \brief Computes the mean of the elements in the array of data
     * 
     * \tparam T data type
     * \tparam TOut type of return output result (default is double)
     * \param idata array of data
     * \param nSize size of the array
     * 
     * \returns The mean of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut mean(T const *idata, const int nSize) const
    {
        return sum<T, TOut>(idata, nSize) / nSize;
    }

    /*!
     * \brief Computes the standard deviation of the elements in the array of data
     * 
     * \tparam T data type
     * \tparam TOut type of return output result (default is double)
     * \param idata array of data
     * \param nSize size of the array
     * 
     * \returns The standard deviation of the elements in the array of data
     */
    template <typename T, typename TOut = double>
    inline TOut stddev(T const *idata, int const nSize, TOut const idatamean = std::numeric_limits<TOut>::max()) const
    {
        TOut m = (idatamean < std::numeric_limits<TOut>::max() ? idatamean : mean<T, TOut>(idata, nSize));
        TOut s(0);
        std::for_each(idata, idata + nSize, [&](T const d) { s += (d - m) * (d - m); });
        return std::sqrt(s / (nSize - 1));
    }

    /*!
     * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold
     * 
     * \tparam T       data type
     * \tparam TOut    data type of return output result (default is double)
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

        TOut weight[fSize];
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
        TOut weightstddev = stddev<TOut, TOut>(weight, fSize, weightmean);

        //return the square of the coefficient of variation (COV)
        return std::pow(weightstddev / weightmean - tol, 2);
    }
};

#endif
