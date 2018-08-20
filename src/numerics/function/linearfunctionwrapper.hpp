#ifndef UMUQ_LINEARFUNCTIONWRAPPER_H
#define UMUQ_LINEARFUNCTIONWRAPPER_H

#include "../function/functiontype.hpp"
#include "../function/umuqdifferentiablefunction.hpp"

/*! \class linearFunctionWrapper
 * \ingroup multimin_Module
 * 
 * \brief Wrapper for an external Multidimensional function
 *  
 * \tparam T Data type
 */
template <typename T>
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
     * \return T 
     */
    T f(T const alpha);

    /*!
     * \brief 
     * 
     * \param alpha 
     * \param DF
     *  
     * \return true 
     * \return false 
     */
    bool df(T const alpha, T *DF);

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
    bool fdf(T const alpha, T *F, T *DF);

    /*!
     * \brief Set the fun object to use the external Multidimensional function
     * 
     * \param umFun  Multidimensional function
     *  
     * \return true 
     * \return false 
     */
    bool set(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>> &umFun);

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
    void prepare(int const n, T const *X, T const F, T const *G, T const *P, T *X_alpha, T *G_alpha);

    /*!
     * \brief 
     * 
     * \param alpha 
     * \param xOut 
     * \param f 
     * \param gOut 
     */
    void updatePosition(T const alpha, T *xOut, T *f, T *gOut);

    /*!
     * \brief 
     * 
     */
    void changeDirection();

    /*!
     * \brief
     *
     */
    void moveTo(T const alpha);

    /*!
     * \brief Calculate the slope along the direction p
     * 
     * \return T 
     */
    inline T slope();

  private:
    //! Multidimensional function
    umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>> fun;

    //! Read only x
    T const *x;

    //! Read only g
    T const *g;

    //! Read only p
    T const *p;

    //! Cached values, for x(alpha) = x + alpha * p
    T f_alpha;
    //!
    T df_alpha;

    //!
    T *x_alpha;
    //!
    T *g_alpha;

    //! Cache "keys"
    T f_cache_key;
    //!
    T df_cache_key;
    //!
    T x_cache_key;
    //!
    T g_cache_key;

    //!
    int nsize;
};

template <typename T>
linearFunctionWrapper<T>::linearFunctionWrapper() {}

template <typename T>
linearFunctionWrapper<T>::~linearFunctionWrapper() {}

template <typename T>
T linearFunctionWrapper<T>::f(T const alpha)
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

template <typename T>
bool linearFunctionWrapper<T>::df(T const alpha, T *DF)
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
            return false;
        }

        g_cache_key = alpha;
    }

    df_alpha = slope();

    df_cache_key = alpha;

    *DF = df_alpha;

    return true;
}

template <typename T>
bool linearFunctionWrapper<T>::fdf(T const alpha, T *F, T *DF)
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

    return false;
}

template <typename T>
bool linearFunctionWrapper<T>::set(umuqDifferentiableFunction<T, F_MTYPE<T>, DF_MTYPE<T>, FDF_MTYPE<T>> &umFun)
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

template <typename T>
void linearFunctionWrapper<T>::prepare(int const n, T const *X, T const F, T const *G, T const *P, T *X_alpha, T *G_alpha)
{
    nsize = n;

    x = X;
    f_alpha = F;
    g = G;
    p = P;

    x_alpha = X_alpha;
    g_alpha = G_alpha;

    x_cache_key = T{};
    f_cache_key = T{};
    g_cache_key = T{};
    df_cache_key = T{};

    std::copy(x, x + nsize, x_alpha);
    std::copy(g, g + nsize, g_alpha);

    df_alpha = slope();
}

template <typename T>
void linearFunctionWrapper<T>::updatePosition(T const alpha, T *X, T *F, T *G)
{
    // Ensure that everything is fully cached
    {
        T F_alpha;
        T DF_alpha;

        fdf(alpha, &F_alpha, &DF_alpha);
    }

    *F = f_alpha;

    std::copy(x_alpha, x_alpha + nsize, X);
    std::copy(g_alpha, g_alpha + nsize, G);
}

template <typename T>
void linearFunctionWrapper<T>::changeDirection()
{
    // Convert the cache values from the end of the current minimization
    // to those needed for the start of the next minimization, alpha=0

    // The new x_alpha for alpha=0 is the current position
    std::copy(x, x + nsize, x_alpha);

    x_cache_key = T{};

    // The function value does not change
    f_cache_key = T{};

    // The new g_alpha for alpha=0 is the current gradient at the endpoint
    std::copy(g, g + nsize, g_alpha);

    g_cache_key = T{};

    // Calculate the slope along the new direction vector, p
    df_alpha = slope();

    df_cache_key = T{};
}

template <typename T>
void linearFunctionWrapper<T>::moveTo(T const alpha)
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

template <typename T>
inline T linearFunctionWrapper<T>::slope()
{
    T s(0);
    for (int i = 0; i < nsize; i++)
    {
        s += g_alpha[i] * p[i];
    }
    return s;
}

#endif // UMUQ_LINEARFUNCTIONWRAPPER
