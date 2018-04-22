#ifndef UMHBM_STATS_H
#define UMHBM_STATS_H

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
        return (TOut)std::accumulate(idata, idata + nSize, (T)0);
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
    inline TOut stddev(T const *idata, const int nSize) const
    {
        TOut m = mean<T, TOut>(idata, nSize);
        TOut s(0);
        std::for_each(idata, idata + nSize, [&](T const d) { s += (d - m) * (d - m); });
        return std::sqrt(s / (nSize - 1));
    }
};

#endif
