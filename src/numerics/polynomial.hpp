#ifndef UMHBM_POLYNOMIAL_H
#define UMHBM_POLYNOMIAL_H

/*! \class polynomial
 *
 * \brief Multivariate monomials with the degree of r in a space of d dimensions.
 *
 *  A (univariate) monomial in 1 variable x is simply any (non-negative integer) power of x:
 *  \f$  1, x, x^2, x^3, \cdots, x^r \f$
 *  The highest exponent of x is termed the degree of the monomial.
 */
template <typename T>
class polynomial
{
  public:
    /*! 
     * \brief Default constructor
     */
    polynomial() : nDim(0), Degree(0) {}

    /*! 
     * \brief constructor
     * 
     * \param dm Dimension (default dimension is 2)
     * \param dg Degree (default degree of accuracy is 2)
     */
    polynomial(unsigned int dm, unsigned int dg) : nDim(dm), Degree(dg) {}

    int binomial_coefficient(int const n, int const k);

    bool monomial_basis(int const d, int const r, int *&alpha);

    int monomial_value(int const d, int const r, int *alpha, T *x, T *&value);
    int monomial_value(int const d, int const r, T *x, T *&value);
    int monomial_value(T *x, T *&value)
    {
        if (nDim == 0 && Degree == 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "There is no Dimesnion and Degree in the constructed object!" << std::endl;
            return 0;
        }
        return monomial_value(nDim, Degree, x, value);
    }

  private:
    int nDim;
    int Degree;

    bool graded_reverse_lexicographic_order(int const d, int const r, int *x);
};

/*! 
 * \brief Computes the binomial coefficient C(n, k).
 *
 * 1) A binomial coefficient C(n, k) can be defined as the coefficient of \f$ X ^ k \f$ in the expansion of \f$ (1 + X) ^ n \f$
 * 2) A binomial coefficient C(n, k) also gives the number of ways, disregarding order, that k objects can be 
 * chosen from among n objects; 
 * more formally, the number of k-element subsets (or k-combinations) of an n-element set.
 * 
 * The formula used is:
 * \f$ c(n,k) = \frac{n!}{ n! * (n-k)! } \f$ 
 * 
 * \param n Input parameter
 * \param k Input parameter
 * 
 * \returns The binomial coefficient \f$ C(n, k) \f$
 */
template <typename T>
int polynomial<T>::binomial_coefficient(int const n, int const k)
{
    if ((k < 0) || (n < 0))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Fatal error! k or n < 0" << std::endl;
        throw(std::runtime_error("Wrong Input!"));
    }
    if (k < n)
    {
        if (k == 0)
        {
            return 1;
        }
        if ((k == 1) || (k == n - 1))
        {
            return n;
        }

        int mn = std::min(k, n - k);
        int mx = std::max(k, n - k);
        int value = mx + 1;
        for (int i = 2; i <= mn; i++)
        {
            value = (value * (mx + i)) / i;
        }

        return value;
    }
    else if (k == n)
    {
        return 1;
    }

    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
    std::cerr << " The binomial coefficient is undefined for k > n " << std::endl;
    return 0;
};

/*! 
 * \brief Use a reverse lexicographic order for next monomial, degree between 0 and r
 *  all monomials in a d dimensional space, with degree r.
 *
 * \param  d   The spatial dimension
 * \param[in]  r   Maximum degree
 * \param[in]  x   Current monomial
 * \param[out] x   Next monomial, last value in the sequence is r.
 */
template <typename T>
bool polynomial<T>::graded_reverse_lexicographic_order(int const d, int const r, int *x)
{
    if (r < 0)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Fatal error! maximum degree r < 0" << std::endl;
        return false;
    }
    if (r == 0)
    {
        return true;
    }

    int asum = std::accumulate(x, x + d, 0);
    if (asum < 0)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Fatal error! input sums < 0" << std::endl;
        return false;
    }

    if (r < asum)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Fatal error! input sums > maximum degree r" << std::endl;
        return false;
    }

    if (x[0] == r)
    {
        x[0] = 0;
        x[d - 1] = 0;
    }
    else
    {
        int i;
        int tmp;

        //Seeking the first index in which x > 0.
        int j = 0;
        for (i = 1; i < d; i++)
        {
            if (x[i] > 0)
            {
                j = i;
                break;
            }
        }

        if (j == 0)
        {
            tmp = x[0];
            x[0] = 0;
            x[d - 1] = tmp + 1;
        }
        else if (j < d - 1)
        {
            x[j] = x[j] - 1;
            tmp = x[0] + 1;
            x[0] = 0;
            x[j - 1] = x[j - 1] + tmp;
        }
        else
        {
            tmp = x[0];
            x[0] = 0;
            x[j - 1] = tmp + 1;
            x[j] = x[j] - 1;
        }
    }
    return true;
};

/*! 
 *   \brief  All monomials in a d dimensional space, with total degree r.
 *   
 *   For example:
 *       d = 2
 *       r = 2
 *
 *       alpha[ 0],[ 1] = 0, 0 = x^0 y^0
 *       alpha[ 2],[ 3] = 1, 0 = x^1 y^0
 *       alpha[ 4],[ 5] = 0, 1 = x^0 y^1
 *       alpha[ 6],[ 7] = 2, 0 = x^2 y^0
 *       alpha[ 8],[ 9] = 1, 1 = x^1 y^1
 *       alpha[10],[11] = 0, 2 = x^0 y^2
 *
 *       monomial_basis(2,2)   = {1,    x,   y,  x^2, xy,  y^2}
 *                       alpha = {0,0, 1,0, 0,1, 2,0, 1,1, 0,2}
 *
 *
 *       monomial_basis(3,2)   = {1,       x,     y,     z,    x^2,  xy,    xz,   y^2,    yz,    z^2  }
 *                       alpha = {0,0,0, 1,0,0, 0,1,0, 0,0,1, 2,0,0 1,1,0, 1,0,1, 0,2,0, 0,1,1, 0,0,2 }
 *
 *
 *   Parameters:
 *       @param[in]  d       The spatial dimension
 *       @param[in]  r       Maximum degree
 *       @param[in]  alpha   Undefined pointer
 *       @param[out] alpha   Pointer to monomial sequence
 */
template <typename T>
bool polynomial<T>::monomial_basis(int const d, int const r, int *&alpha)
{
    if (alpha != nullptr)
    {
        delete[] alpha;
    }

    try
    {
        int const N = d * polynomial<T>::binomial_coefficient(d + r, r);
        alpha = new int[N];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        return false;
    }

    int x[d] = {};
    int n = 0;

    for (;;)
    {
        for (int j = d - 1; j >= 0; j--, n++)
        {
            alpha[n] = x[j];
        }

        if (x[0] == r)
        {
            return true;
        }

        if (!polynomial<T>::graded_reverse_lexicographic_order(d, r, x))
        {
            return false;
        }
    }
    return true;
};

/*! \brief Evaluates a monomial at a point x.
 *
 * @param[in]  d       The spatial dimension
 * @param[in]  r       Maximum degree
 * @param[in]  alpha   The exponents of the monomial
 * @param[in]  x       The coordinates of the evaluation points
 * @param[out] value   Monomial_value, the array value of the monomial at point x
 * 
 * \returns the size of the monomial array
 */
template <typename T>
int polynomial<T>::monomial_value(int const d, int const r, int *alpha, T *x, T *&value)
{
    int const n = polynomial<T>::binomial_coefficient(d + r, r);

    if (value == nullptr)
    {
        try
        {
            value = new T[n];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return 0;
        }
    }

    for (int i = 0, k = 0; i < n; i++)
    {
        T v = static_cast<T>(1);
        for (int j = 0; j < d; j++, k++)
        {
            v *= std::pow(x[j], alpha[k]);
        }
        value[i] = v;
    }

    return n;
};

/*! 
 * \brief Evaluates a monomial at a point x.
 *
 * @param[in]  d       The dimension
 * @param[in]  r       Maximum degree
 * @param[in]  x       The coordinates of the evaluation point
 * @param[out] value   Monomial_value, the array value of the monomial at point x
 * 
 * \returns the size of the monomial array
 */
template <typename T>
int polynomial<T>::monomial_value(int const d, int const r, T *x, T *&value)
{
    int const n = polynomial<T>::binomial_coefficient(d + r, r);

    if (value == nullptr)
    {
        try
        {
            value = new T[n];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        }
    }

    int *alpha = nullptr;

    if (polynomial<T>::monomial_basis(d, r, alpha))
    {

        for (int i = 0, k = 0; i < n; i++)
        {
            T v = static_cast<T>(1);
            for (int j = 0; j < d; j++, k++)
            {
                v *= std::pow(x[j], alpha[k]);
            }
            value[i] = v;
        }

        delete[] alpha;
        return n;
    }

    delete[] value;
    value = nullptr;
    return 0;
};

#endif
