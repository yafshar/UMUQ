#ifndef UMUQ_LINEARFUNCTIONWRAPPER_H
#define UMUQ_LINEARFUNCTIONWRAPPER_H

#include "core/core.hpp"
#include "datatype/functiontype.hpp"
#include "umuqdifferentiablefunction.hpp"

#include <algorithm>

namespace umuq
{

inline namespace multimin
{

/*! \class linearFunctionWrapper
 * \ingroup Multimin_Module
 *
 * \brief Wrapper for an external Multidimensional function
 *
 * \tparam DataType Data type
 */
template <typename DataType>
class linearFunctionWrapper
{
  public:
    /*!
     * \brief Construct a new linear Function Wrapper object
     *
     */
    linearFunctionWrapper();

    /*!
     * \brief Destroy the linear Function Wrapper object
     *
     */
    ~linearFunctionWrapper();

    /*!
     * \brief Linear function
     *
     * \param alpha Input data
     *
     * \return DataType
     */
    DataType f(DataType const alpha);

    /*!
     * \brief
     *
     * \param alpha
     * \param DF
     *
     * \return true
     * \return false
     */
    bool df(DataType const alpha, DataType *DF);

    /*!
     * \brief
     *
     * \param alpha
     * \param F
     * \param DF
     *
     * \return true
     * \return false
     */
    bool fdf(DataType const alpha, DataType *F, DataType *DF);

    /*!
     * \brief Set the fun object to use the external Multidimensional function
     *
     * \param umFun  Multidimensional function
     *
     * \return true
     * \return false
     */
    bool set(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>> &umFun);

    /*!
     * \brief
     *
     * \param n
     * \param X
     * \param F
     * \param G
     * \param P
     * \param X_alpha
     * \param G_alpha
     */
    void prepare(int const n, DataType const *X, DataType const F, DataType const *G, DataType const *P, DataType *X_alpha, DataType *G_alpha);

    /*!
     * \brief
     *
     * \param alpha
     * \param xOut
     * \param f
     * \param gOut
     */
    void updatePosition(DataType const alpha, DataType *xOut, DataType *f, DataType *gOut);

    /*!
     * \brief
     *
     */
    void changeDirection();

    /*!
     * \brief
     *
     */
    void moveTo(DataType const alpha);

    /*!
     * \brief Calculate the slope along the direction p
     *
     * \return DataType
     */
    inline DataType slope();

  private:
    /*!
     * \brief Delete a linearFunctionWrapper object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    linearFunctionWrapper(linearFunctionWrapper<DataType> const &) = delete;

    /*!
     * \brief Delete a linearFunctionWrapper object assignment
     *
     * Avoiding implicit copy assignment.
     *
     * \returns linearFunctionWrapper<DataType>&
     */
    linearFunctionWrapper<DataType> &operator=(linearFunctionWrapper<DataType> const &) = delete;

  private:
    //! Multidimensional function
    umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>> fun;

    //! Read only x
    DataType const *x;

    //! Read only g
    DataType const *g;

    //! Read only p
    DataType const *p;

    //! Cached values, for x(alpha) = x + alpha * p
    DataType f_alpha;
    //!
    DataType df_alpha;

    //!
    DataType *x_alpha;
    //!
    DataType *g_alpha;

    //! Cache "keys"
    DataType f_cache_key;
    //!
    DataType df_cache_key;
    //!
    DataType x_cache_key;
    //!
    DataType g_cache_key;

    //!
    int nsize;
};

template <typename DataType>
linearFunctionWrapper<DataType>::linearFunctionWrapper() {}

template <typename DataType>
linearFunctionWrapper<DataType>::~linearFunctionWrapper() {}

template <typename DataType>
DataType linearFunctionWrapper<DataType>::f(DataType const alpha)
{
    // Using previously cached f(alpha)
    if (alpha == f_cache_key)
    {
        return f_alpha;
    }

    moveTo(alpha);

    f_alpha = fun.f(x_alpha);

    f_cache_key = alpha;

    return f_alpha;
}

template <typename DataType>
bool linearFunctionWrapper<DataType>::df(DataType const alpha, DataType *DF)
{
    // Using previously cached df(alpha)
    if (alpha == df_cache_key)
    {
        *DF = df_alpha;

        return true;
    }

    moveTo(alpha);

    if (alpha != g_cache_key)
    {
        if (!fun.df(x_alpha, g_alpha))
        {
            UMUQFAILRETURN("Failed to compute the gradient of f!");
        }

        g_cache_key = alpha;
    }

    df_alpha = slope();

    df_cache_key = alpha;

    *DF = df_alpha;

    return true;
}

template <typename DataType>
bool linearFunctionWrapper<DataType>::fdf(DataType const alpha, DataType *F, DataType *DF)
{
    // Check for previously cached values
    if (alpha == f_cache_key && alpha == df_cache_key)
    {
        *F = f_alpha;
        *DF = df_alpha;

        return true;
    }

    if (alpha == f_cache_key || alpha == df_cache_key)
    {
        *F = f(alpha);

        return df(alpha, DF);
    }

    moveTo(alpha);

    if (fun.fdf(x_alpha, &f_alpha, g_alpha))
    {
        f_cache_key = alpha;
        g_cache_key = alpha;

        df_alpha = slope();
        df_cache_key = alpha;

        *F = f_alpha;
        *DF = df_alpha;

        return true;
    }

    UMUQFAILRETURN("Failed to compute the function f and its gradient!");
}

template <typename DataType>
bool linearFunctionWrapper<DataType>::set(umuqDifferentiableFunction<DataType, F_MTYPE<DataType>, DF_MTYPE<DataType>, FDF_MTYPE<DataType>> &umFun)
{
    if (umFun)
    {
        fun.f = umFun.f;
        fun.df = umFun.df;
        fun.fdf = umFun.fdf;

        return true;
    }

    UMUQFAILRETURN("Function is not assigned!");
}

template <typename DataType>
void linearFunctionWrapper<DataType>::prepare(int const n, DataType const *X, DataType const F, DataType const *G, DataType const *P, DataType *X_alpha, DataType *G_alpha)
{
    nsize = n;

    x = X;
    f_alpha = F;
    g = G;
    p = P;

    x_alpha = X_alpha;
    g_alpha = G_alpha;

    x_cache_key = DataType{};
    f_cache_key = DataType{};
    g_cache_key = DataType{};
    df_cache_key = DataType{};

    std::copy(x, x + nsize, x_alpha);
    std::copy(g, g + nsize, g_alpha);

    df_alpha = slope();
}

template <typename DataType>
void linearFunctionWrapper<DataType>::updatePosition(DataType const alpha, DataType *X, DataType *F, DataType *G)
{
    // Ensure that everything is fully cached
    {
        DataType F_alpha;
        DataType DF_alpha;

        fdf(alpha, &F_alpha, &DF_alpha);
    }

    *F = f_alpha;

    std::copy(x_alpha, x_alpha + nsize, X);
    std::copy(g_alpha, g_alpha + nsize, G);
}

template <typename DataType>
void linearFunctionWrapper<DataType>::changeDirection()
{
    // Convert the cache values from the end of the current minimization
    // to those needed for the start of the next minimization, alpha=0

    // The new x_alpha for alpha=0 is the current position
    std::copy(x, x + nsize, x_alpha);

    x_cache_key = DataType{};

    // The function value does not change
    f_cache_key = DataType{};

    // The new g_alpha for alpha=0 is the current gradient at the endpoint
    std::copy(g, g + nsize, g_alpha);

    g_cache_key = DataType{};

    // Calculate the slope along the new direction vector, p
    df_alpha = slope();

    df_cache_key = DataType{};
}

template <typename DataType>
void linearFunctionWrapper<DataType>::moveTo(DataType const alpha)
{
    // Using previously cached position
    if (alpha == x_cache_key)
    {
        return;
    }

    x_cache_key = alpha;

    // Set \f$ x_alpha = x + alpha * p \f$
    std::copy(x, x + nsize, x_alpha);

    for (int i = 0; i < nsize; i++)
    {
        x_alpha[i] += alpha * p[i];
    }
}

template <typename DataType>
inline DataType linearFunctionWrapper<DataType>::slope()
{
    DataType s(0);
    for (int i = 0; i < nsize; i++)
    {
        s += g_alpha[i] * p[i];
    }
    return s;
}

} // namespace multimin
} // namespace umuq

#endif // UMUQ_LINEARFUNCTIONWRAPPER
