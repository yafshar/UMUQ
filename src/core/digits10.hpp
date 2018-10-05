#ifndef UMUQ_DIGITS10_H
#define UMUQ_DIGITS10_H

/*!
 * \ingroup Core_Module
 * 
 * \brief Default digits10() class based on std::numeric_limits
 * 
 * \tparam T data type
 */
template <typename T,
          bool use_numeric_limits = std::numeric_limits<T>::is_specialized,
          bool is_integer = std::numeric_limits<T>::is_integer>
struct default_digits10
{
    static int run() { return std::numeric_limits<T>::digits10; }
};

/*!
 * \ingroup Core_Module
 * 
 * \brief Specialization for Floating point
 * 
 * \tparam T data type
 */
template <typename T>
struct default_digits10<T, false, false>
{
    static int run() { return int(std::ceil(-std::log10(std::numeric_limits<T>::epsilon()))); }
};

/*!
 * \ingroup Core_Module
 * 
 * \brief Specialization for Integer
 * 
 * \tparam T data type
 */
template <typename T>
struct default_digits10<T, false, true>
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
 * \tparam T data type
 */
template <typename T>
static inline int digits10()
{
    return default_digits10<T>::run();
}

#endif // UMUQ_DIGITS10
