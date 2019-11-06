#ifndef UMUQ_DIGITS10_H
#define UMUQ_DIGITS10_H

#include <cmath>

#include <limits>

/*!
 * \ingroup Core_Module
 *
 * \brief Default digits10() class based on std::numeric_limits
 *
 * \tparam DataType data type
 */
template <typename DataType,
          bool use_numeric_limits = std::numeric_limits<DataType>::is_specialized,
          bool is_integer = std::numeric_limits<DataType>::is_integer>
struct default_digits10
{
    static int run() { return std::numeric_limits<DataType>::digits10; }
};

/*!
 * \ingroup Core_Module
 *
 * \brief Specialization for Floating point
 *
 * \tparam DataType data type
 */
template <typename DataType>
struct default_digits10<DataType, false, false>
{
    static int run() { return int(std::ceil(-std::log10(std::numeric_limits<DataType>::epsilon()))); }
};

/*!
 * \ingroup Core_Module
 *
 * \brief Specialization for Integer
 *
 * \tparam DataType data type
 */
template <typename DataType>
struct default_digits10<DataType, false, true>
{
    static int run() { return 0; }
};

/*!
 * \ingroup Core_Module
 *
 * \brief digits10(), implementation
 *
 * It is based on std::numeric_limits if specialized, 0 for integer types, and
 * log10(epsilon()) otherwise.
 *
 * \tparam DataType data type
 */
template <typename DataType>
static inline int digits10()
{
    return default_digits10<DataType>::run();
}

#endif // UMUQ_DIGITS10
