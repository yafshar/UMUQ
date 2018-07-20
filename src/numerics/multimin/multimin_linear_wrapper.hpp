#ifndef UMUQ_MULTIMIN_LINEAR_WRAPPER_H
#define UMUQ_MULTIMIN_LINEAR_WRAPPER_H

/*! \class linearFunctionWrapper
 * \ingroup multimin_Module
 * 
 * \brief 
 *  
 * \tparam T      Data type
 * \tparam TMFD   Multimin differentiable function type
 */
template <typename T, class TMFD>
class linearFunctionWrapper
{
  public:
    linearFunctionWrapper();

    /*!
     * \brief 
     * 
     */
    void prepare(TMFD *fdf, T const *xIn, T const f, T const *gIn, T const *p_, T *x_alpha_, T *g_alpha_);

    /*!
     * \brief 
     * 
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
     * \brief Compute gradient direction
     */
    inline T slope();

  public:
    /*! \class functionfdfWrapper
     * \brief Wrapper for function_fdf
     * 
     */
    class functionfdfWrapper : public function_fdf<T, functionfdfWrapper>
    {
      public:
        /*!
         * \brief Construct a new functionfdf Wrapper object
         * 
         * \param linearFunctionWrapperRef 
         */
        explicit functionfdfWrapper(linearFunctionWrapper<T, TMFD> &linearFunctionWrapperRef);

        /*!
         * \brief 
         * 
         * \param alpha 
         * \return T 
         */
        T f(T const alpha);

        /*!
         * \brief 
         * 
         * \param alpha 
         * \return T 
         */
        T df(T const alpha);

        /*!
         * \brief 
         * 
         * \param alpha 
         * \param f 
         * \param df 
         */
        void fdf(T const alpha, T *f, T *df);

      private:
        //!
        linearFunctionWrapper<T, TMFD> &lfw;
    };

  private:
    //!
    TMFD *wfdf;
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
    std::size_t nsize;

  public:
    //!
    functionfdfWrapper fdf_linear;
};

template <typename T, class TMFD>
linearFunctionWrapper<T, TMFD>::functionfdfWrapper::functionfdfWrapper(linearFunctionWrapper<T, TMFD> &linearFunctionWrapperRef) : lfw(linearFunctionWrapperRef)
{
}

template <typename T, class TMFD>
T linearFunctionWrapper<T, TMFD>::functionfdfWrapper::f(T const alpha)
{
    //using previously cached f(alpha)
    if (alpha == this->lfw.f_cache_key)
    {
        return this->lfw.f_alpha;
    }

    this->lfw.moveTo(alpha);
    this->lfw.f_alpha = this->lfw.wfdf->f(this->lfw.x_alpha);
    this->lfw.f_cache_key = alpha;

    return this->lfw.f_alpha;
}

template <typename T, class TMFD>
T linearFunctionWrapper<T, TMFD>::functionfdfWrapper::df(T const alpha)
{
    //using previously cached df(alpha)
    if (alpha == this->lfw.df_cache_key)
    {
        return this->lfw.df_alpha;
    }

    this->lfw.moveTo(alpha);

    if (alpha != this->lfw.g_cache_key)
    {
        this->lfw.wfdf->df(this->lfw.x_alpha, this->lfw.g_alpha);
        this->lfw.g_cache_key = alpha;
    }

    this->lfw.df_alpha = this->lfw.slope();
    this->lfw.df_cache_key = alpha;

    return this->lfw.df_alpha;
}

template <typename T, class TMFD>
void linearFunctionWrapper<T, TMFD>::functionfdfWrapper::fdf(T const alpha, T *f, T *df)
{
    //Check for previously cached values
    if (alpha == this->lfw.f_cache_key && alpha == this->lfw.df_cache_key)
    {
        *f = this->lfw.f_alpha;
        *df = this->lfw.df_alpha;
        return;
    }
    if (alpha == this->lfw.f_cache_key || alpha == this->lfw.df_cache_key)
    {
        *f = this->lfw.fdf_linear.f(alpha);
        *df = this->lfw.fdf_linear.df(alpha);
        return;
    }

    this->lfw.moveTo(alpha);
    this->lfw.wfdf->fdf(this->lfw.x_alpha, &this->lfw.f_alpha, this->lfw.g_alpha);
    this->lfw.f_cache_key = alpha;
    this->lfw.g_cache_key = alpha;
    this->lfw.df_alpha = this->lfw.slope();
    this->lfw.df_cache_key = alpha;

    *f = this->lfw.f_alpha;
    *df = this->lfw.df_alpha;
}

template <typename T, class TMFD>
linearFunctionWrapper<T, TMFD>::linearFunctionWrapper() : fdf_linear(*this) {}

template <typename T, class TMFD>
void linearFunctionWrapper<T, TMFD>::prepare(TMFD *fdf, T const *xIn, T const f, T const *gIn, T const *p_, T *x_alpha_, T *g_alpha_)
{
    this->wfdf = fdf;
    this->nsize = this->wfdf->n;
    this->fdf_linear.n = this->nsize;

    this->x = xIn;
    this->f_alpha = f;
    this->g = gIn;
    this->p = p_;
    this->x_alpha = x_alpha_;
    this->g_alpha = g_alpha_;

    this->x_cache_key = T{};
    this->f_cache_key = T{};
    this->g_cache_key = T{};
    this->df_cache_key = T{};

    std::copy(this->x, this->x + this->nsize, this->x_alpha);
    std::copy(this->g, this->g + this->nsize, this->g_alpha);

    this->df_alpha = this->slope();
}

/*!
     * \brief 
     * 
     */
template <typename T, class TMFD>
void linearFunctionWrapper<T, TMFD>::updatePosition(T const alpha, T *xOut, T *f, T *gOut)
{
    //Ensure that everything is fully cached
    {
        T f_alpha_;
        T df_alpha_;

        this->fdf_linear.fdf(alpha, &f_alpha_, &df_alpha_);
    }

    *f = this->f_alpha;

    std::copy(this->x_alpha, this->x_alpha + this->nsize, xOut);
    std::copy(this->g_alpha, this->g_alpha + this->nsize, gOut);
}

/*!
     * \brief 
     * 
     */
template <typename T, class TMFD>
void linearFunctionWrapper<T, TMFD>::changeDirection()
{
    //Convert the cache values from the end of the current minimization
    //to those needed for the start of the next minimization, alpha=0

    //The new x_alpha for alpha=0 is the current position
    std::copy(this->x, this->x + this->nsize, this->x_alpha);

    this->x_cache_key = T{};

    //The function value does not change
    this->f_cache_key = T{};

    //The new g_alpha for alpha=0 is the current gradient at the endpoint
    std::copy(this->g, this->g + this->nsize, this->g_alpha);

    this->g_cache_key = T{};

    //Calculate the slope along the new direction vector, p
    this->df_alpha = this->slope();

    this->df_cache_key = T{};
}

/*!
     * \brief 
     * 
     */
template <typename T, class TMFD>
void linearFunctionWrapper<T, TMFD>::moveTo(T const alpha)
{
    //using previously cached position
    if (alpha == this->x_cache_key)
    {
        return;
    }
    this->x_cache_key = alpha;

    //Set \f$ x_alpha = x + alpha * p \f$
    std::copy(this->x, this->x + this->nsize, this->x_alpha);

    for (std::size_t i = 0; i < this->nsize; i++)
    {
        this->x_alpha[i] += alpha * this->p[i];
    }
}

/*!
     * \brief Compute gradient direction
     */
template <typename T, class TMFD>
inline T linearFunctionWrapper<T, TMFD>::slope()
{
    T tmp(0);
    for (std::size_t i = 0; i < this->nsize; i++)
    {
        tmp += this->g_alpha[i] * this->p[i];
    }
    return tmp;
}

#endif
