#ifndef UMUQ_META_H
#define UMUQ_META_H

#include <type_traits>

/*!
 * \ingroup Core_Module
 *
 * \brief Provides member typedef type, which is defined as Then if Condition is true at compile time, or as Else if Condition is false.
 *
 * \tparam Condition  Condition
 * \tparam Then       typedef Then
 * \tparam Else       typedef Else
 */
template <bool Condition, typename Then, typename Else>
struct conditional
{
    typedef Then type;
};

template <typename Then, typename Else>
struct conditional<false, Then, Else>
{
    typedef Else type;
};

/*!
 * \ingroup Core_Module
 *
 * \brief If T and U name the same type, provides the member constant value equal to true. Otherwise value is false.
 *
 * \tparam T type T
 * \tparam U type U
 */
template <typename T, typename U>
struct is_same
{
    enum
    {
        value = 0
    };
};

template <typename T>
struct is_same<T, T>
{
    enum
    {
        value = 1
    };
};

/*!
 * \ingroup Core_Module
 *
 * \brief Checks whether \c T is a floating-point type. Provides the member constant value
 * which is equal to \c true, if \c T is any of the type \c float, \c double, \c long double.
 * Otherwise, \c value is equal to \c false.
 *
 * \tparam T Data type
 */
template <class T>
struct umuq_float : std::integral_constant<bool, std::is_same<float, T>::value ||
                                                     std::is_same<double, T>::value ||
                                                     std::is_same<long double, T>::value>
{
};

/*!
 * \brief Checks whether \c T is an integer type. Provides the member constant value
 * which is equal to \c true, if \c T is any of the type \c short, \c unsigned short,
 * \c int, \c unsigned int, \c long, \c unsigned long, \c long long, and
 * \c unsigned long long. Otherwise, \c value is equal to \c false.
 *
 * \tparam T
 */
template <class T>
struct umuq_int : std::integral_constant<bool, std::is_same<short, T>::value || std::is_same<unsigned short, T>::value ||
                                                   std::is_same<int, T>::value || std::is_same<unsigned int, T>::value ||
                                                   std::is_same<long, T>::value || std::is_same<unsigned long, T>::value ||
                                                   std::is_same<long long, T>::value || std::is_same<unsigned long long, T>::value>
{
};

/*!
 * \ingroup Core_Module
 *
 * It computes int(sqrt(\a Y)) with \a Y an integer.
 * Usage example: \code meta_sqrt<1023>::ret \endcode
 */
template <int Y,
          int InfX = 0,
          int SupX = ((Y == 1) ? 1 : Y / 2),
          bool Done = ((SupX - InfX) <= 1 ? true : ((SupX * SupX <= Y) && ((SupX + 1) * (SupX + 1) > Y)))>
class meta_sqrt
{
    enum
    {
        MidX = (InfX + SupX) / 2,
        TakeInf = MidX * MidX > Y ? 1 : 0,
        NewInf = int(TakeInf) ? InfX : int(MidX),
        NewSup = int(TakeInf) ? int(MidX) : SupX
    };

public:
    enum
    {
        ret = meta_sqrt<Y, NewInf, NewSup>::ret
    };
};

template <int Y, int InfX, int SupX>
class meta_sqrt<Y, InfX, SupX, true>
{
public:
    enum
    {
        ret = (SupX * SupX <= Y) ? SupX : InfX
    };
};

/*!
 * \ingroup Core_Module
 *
 * Computes the least common multiple of two positive integer A and B
 * at compile-time. It implements a naive algorithm testing all multiples of A.
 * It thus works better if A>=B.
 */
template <int A, int B, int K = 1, bool Done = ((A * K) % B) == 0>
struct meta_least_common_multiple
{
    enum
    {
        ret = meta_least_common_multiple<A, B, K + 1>::ret
    };
};

template <int A, int B, int K>
struct meta_least_common_multiple<A, B, K, true>
{
    enum
    {
        ret = A * K
    };
};

/*!
 * \ingroup Core_Module
 *
 * \brief Computes \f$ A^N \f$ mod \f$ 2^32 \f$
 */
template <unsigned int A, unsigned int N>
struct CTpow
{
    static const unsigned int value = (N & 1 ? A : 1) * CTpow<A * A, N / 2>::value;
};

/*!
 * \ingroup Core_Module
 *
 * \brief Specialization to terminate recursion: \f$ A^0 = 1 \f$
 */
template <unsigned int A>
struct CTpow<A, 0>
{
    static const unsigned int value = 1;
};

/*!
 * \ingroup Core_Module
 *
 * CTpowseries<A,N> computes \f$ 1+A+A^2+A^3+A^4+A^5 \cdots + A^(N-1) \f$ mod \f$ 2^32 \f$.
 * We do NOT use the more elegant formula \f$ \frac{(a^N-1)}{(a-1)} \f$ (see Knuth
 * 3.2.1), because it's more awkward to compute with implicit mod \f$ 2^32 \f$.
 * Based on recursion:
 *
 * \verbatim
 * g(A,n)= (1+A)*g(A*A, n/2);      if n is even
 * g(A,n)= 1+A*(1+A)*g(A*A, n/2);  if n is ODD (since n/2 truncates)
 * \endverbatim
 */
template <unsigned int A, unsigned int N>
struct CTpowseries
{
    static const unsigned int recurse = (1 + A) * CTpowseries<A * A, N / 2>::value;
    static const unsigned int value = (N & 1) ? 1 + A * recurse : recurse;
};

template <unsigned int A>
struct CTpowseries<A, 0>
{
    static const unsigned int value = 0;
};

template <unsigned int A>
struct CTpowseries<A, 1>
{
    static const unsigned int value = 1;
};

/*!
 * \ingroup Core_Module
 *
 * \brief Compute A*B mod m.  Tricky only because of implicit \f$ 2^32 \f$ modulus.
 * Uses recursion.
 *
 * \verbatim
 * if A is even, then A*B mod m =   (A/2)*(B+B mod m) mod m.
 * if A is odd,  then A*B mod m =  ((A/2)*(B+B mod m) mod m) + B mod m.
 * \endverbatim
 */
template <unsigned int A, unsigned int B, unsigned int m>
struct CTmultmod
{
    // (A/2)*(B*2) mod m
    static const unsigned int temp = CTmultmod<A / 2, (B >= m - B ? B + B - m : B + B), m>::value;
    static const unsigned int value = A & 1 ? ((B >= m - temp) ? B + temp - m : B + temp) : temp;
};

/*!
 * \ingroup Core_Module
 *
 * \brief Specialization to terminate the recursion
 */
template <unsigned int B, unsigned int m>
struct CTmultmod<0, B, m>
{
    static const unsigned int value = 0;
};

#endif // UMUQ_META
