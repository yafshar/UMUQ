#ifndef UMHBM_POLYNOMIAL_H
#define UMHBM_POLYNOMIAL_H

/*! \brief Multivariate monomials with the degree of r in a space of d dimensions.
 *
 *  A (univariate) monomial in 1 variable x is simply any (non-negative integer) power of x:
 *      1, x, x^2, x^3, ..., x^r
 *  The highest exponent of x is termed the degree of the monomial.
 */
class polynomial
{
  public:
    polynomial() : dim(2), degree(2){};
    inline polynomial(unsigned int dm);
    inline polynomial(unsigned int dm, unsigned int dg);

    template <class T>
    inline T const min_value(T const a, T const b);
    template <class T>
    inline T const max_value(T const a, T const b);

    template <typename T>
    inline T arraysum(int const arraysize, T const *array);

    int binomial_coefficient(int const n, int const k);

    void monomial_basis(int const d, int const r, int *&alpha);

    template <typename T>
    void monomial_value(int const d, int const r, int *alpha, T *x, T *&value);

  private:
    unsigned int dim;
    unsigned int degree;

    void graded_reverse_lexicographic_order(int const d, int const r, int *x);
};

inline polynomial::polynomial(unsigned int dm)
{
    polynomial::dim = dm;
    polynomial::degree = 2;
};

inline polynomial::polynomial(unsigned int dm, unsigned int dg)
{
    polynomial::dim = dm;
    polynomial::degree = dg;
};

/*! \fn min_value(a, b)
*   \brief A function that returns the minimum of \a a and \a b.
*/
template <typename T>
inline T const polynomial::min_value(T const a, T const b)
{
    if (a > b)
        return b;
    return a;
};

/*! \fn max_value(a, b)
*   \brief A function that returns the maximum of \a a and \a b.
*/
template <typename T>
inline T const polynomial::max_value(T const a, T const b)
{
    if (b > a)
        return b;
    return a;
};

/*! \fn arraysum(arraysize, array)
*   \brief A function that returns the sum of an array of arraysize.
*/
template <typename T>
inline T polynomial::arraysum(int const arraysize, T const *array)
{
    return std::accumulate(array, array + arraysize, 0);
};

/*! 
 * \brief Computes the binomial coefficient C(n, k).
 *
 * 1) A binomial coefficient C(n, k) can be defined as the coefficient of X ^ k in the expansion of(1 + X) ^ n
 * 2) A binomial coefficient C(n, k) also gives the number of ways, disregarding order, that k objects can be 
 * chosen from among n objects; 
 * more formally, the number of k-element subsets (or k-combinations) of an n-element set.
 * 
 * The formula used is:
 *   c(n,k) = n! / ( n! * (n-k)! )  
 *  $c(n,k) = \frac{n\!}{( n\! * (n-k)! )}$ 
 * 
 * @param[in] n
 * @param[in] k
 * @param[out] binomial The binomial coefficient
 */
int polynomial::binomial_coefficient(int const n, int const k)
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
            return 1;
        if ((k == 1) || (k == n - 1))
            return n;
        int mn;
        int mx;
        int value;
        mn = polynomial::min_value(k, n - k);
        mx = polynomial::max_value(k, n - k);
        value = mx + 1;
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
 *   \brief Use a reverse lexicographic order for next monomial, degree between 0 and r
 *   all monomials in a d dimensional space, with degree r.
 *
 *   Parameters:
 *       @param[in]  d   The spatial dimension
 *       @param[in]  r   Maximum degree
 *       @param[in]  x   Current monomial
 *       @param[out] x   Next monomial, last value in the sequence is r.
 */
void polynomial::graded_reverse_lexicographic_order(int const d, int const r, int *x)
{
    if (r < 0)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Fatal error! maximum degree r < 0" << std::endl;
        throw(std::runtime_error("Wrong Input!"));
    }
    if (r == 0)
        return;

    int asum;
    asum = polynomial::arraysum(d, x);

    if (asum < 0)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Fatal error! input sums < 0" << std::endl;
        throw(std::runtime_error("Wrong Input!"));
    }

    if (r < asum)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Fatal error! input sums > maximum degree r" << std::endl;
        throw(std::runtime_error("Wrong Input!"));
    }

    if (x[0] == r)
    {
        x[0] = 0;
        x[d - 1] = 0;
    }
    else
    {
        int i;
        int j;
        int tmp;

        // Seeking the first index in which x > 0.
        j = 0;
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
    return;
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
void polynomial::monomial_basis(int const d, int const r, int *&alpha)
{
    int j;
    int n;

    if (alpha != NULL)
    {
        delete[] alpha;
    }

    n = d * polynomial::binomial_coefficient(d + r, r);
    try
    {
        alpha = new int[n];
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
    }

    int x[d];
    for (int i = 0; i < d; i++)
    {
        x[i] = 0;
    };

    n = 0;
    for (;;)
    {
        for (j = d - 1; j >= 0; j--, n++)
        {
            alpha[n] = x[j];
        }

        if (x[0] == r)
            break;

        polynomial::graded_reverse_lexicographic_order(d, r, x);
    }
    return;
};

/*! \brief Evaluates a monomial at a point x.
 *
 *   Parameters:
 *       @param[in]  d       The spatial dimension
 *       @param[in]  r       Maximum degree
 *       @param[in]  alpha   The exponents of the monomial
 *       @param[in]  x       The coordinates of the evaluation points
 *       @param[out] value   Monomial_value, the array value of the monomial at point x
 */
template <typename T>
void polynomial::monomial_value(int const d, int const r, int *alpha, T *x, T *&value)
{
    const int n = polynomial::binomial_coefficient(d + r, r);

    T v;
    int k;

    k = 0;
    for (int i = 0; i < n; i++)
    {
        v = 1;
        for (int j = 0; j < d; j++, k++)
        {
            v *= std::pow(x[j], alpha[k]);
        }
        value[i] = v;
    }
};

#endif
