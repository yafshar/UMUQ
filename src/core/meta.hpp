#ifndef UMHBM_META_H
#define UMHBM_META_H

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

/*! In short, it computes int(sqrt(\a Y)) with \a Y an integer.
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

#endif 
