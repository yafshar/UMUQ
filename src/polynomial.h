#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

class polynomial
{

  public:
    template <class T>
    inline T min_value(T a, T b);
    template <class T>
    inline T max_value(T a, T b);
    template <class T>
    inline T arraysum(int arraysize, T *array);

    int binomial_coefficient(int n, int k)
    {
        //******************************************************************************/
        //   Purpose:
        //       computes the binomial coefficient c(n, k).
        //   The formula used is:
        //       c(n,k) = n! / ( n! * (n-k)! )
        //   Parameters:
        //       Input : values of n and k.
        //       Output: binomial
        //******************************************************************************/
        if (k > n)
            return 0;
        if (k == n)
            return 1;
        if (k == 0)
            return 1;
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

    void graded_reverse_lexicographic_order(int d, int r, int *x)
    {
        //*****************************************************************************/
        //   Purpose:
        //     uses a reverse lexicographic order for next monomial, degree between 0 and r
        //     all monomials in a d dimensional space, with degree r
        //   Parameters:
        //     Input : d, the spatial dimension
        //             r, maximum degree
        //             x, current monomial
        //     Output: x, next monomial, last value in the sequence is r.
        //*****************************************************************************/
        if (r < 0)
        {
            std::cout << std::endl;
            std::cout << " Fatal error! maximum degree r < 0" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (r == 0)
            return;

        int asum;
        asum = polynomial::arraysum(d, x);

        if (asum < 0)
        {
            std::cout << std::endl;
            std::cout << " Fatal error! input sums < 0" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (r < asum)
        {
            std::cout << std::endl;
            std::cout << " Fatal error! input sums > maximum degree r" << std::endl;
            exit(EXIT_FAILURE);
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
    }

    void monomial_basis(int d, int r, int *&alpha)
    {
        //******************************************************************************/
        //   Purpose:
        //     All monomials in a d dimensional space, with total degree r.
        //   For example:
        //       d = 2
        //       r = 2
        //
        //       alpha[ 0],[ 1] = 0, 0 = x^0 y^0
        //       alpha[ 2],[ 3] = 1, 0 = x^1 y^0
        //       alpha[ 4],[ 5] = 0, 1 = x^0 y^1
        //       alpha[ 6],[ 7] = 2, 0 = x^2 y^0
        //       alpha[ 8],[ 9] = 1, 1 = x^1 y^1
        //       alpha[10],[11] = 0, 2 = x^0 y^2
        //
        //       monomial_basis(2,2) = {1, x, y, x^2, xy, y^2}
        //
        //   Parameters:
        //     Input : d,     the spatial dimension
        //             r,     maximum degree
        //             alpha, undefined pointer
        //     Output: alpha, pointer to monomial sequence
        //******************************************************************************/
        int j;
        int n;

        if (alpha == NULL)
        {
            n = d * polynomial::binomial_coefficient(d + r, r);
            try
            {
                alpha = new int[n];
            }
            catch (const std::system_error &e)
            {
                std::cout << " System error with code " << e.code() << " meaning " << e.what() << std::endl;
            }
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
    }

    template <class T>
    T *monomial_value(int d, int r, int *alpha, T *x)
    {
        //******************************************************************************/
        //   Purpose:
        //     Evaluates a monomial at a point x
        //   Parameters:
        //     Input : d,     the spatial dimension
        //             r,     maximum degree
        //             alpha, the exponents of the monomial
        //             x,     the coordinates of the evaluation points
        //     Output: monomial_value, the array value of the monomial at point x
        //******************************************************************************/
        int n;
        T *value = NULL;

        n = polynomial::binomial_coefficient(d + r, r);
        value = new T[n];

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
        return value;
    }

  private:
};

template <class T>
inline T polynomial::min_value(T a, T b)
{
    if (a > b)
        return b;
    return a;
}

template <class T>
inline T polynomial::max_value(T a, T b)
{
    if (b > a)
        return b;
    return a;
}

template <class T>
inline T polynomial::arraysum(int arraysize, T *array)
{
    T sum;
    sum = 0;
    for (int i = 0; i < arraysize; i++)
    {
        sum += array[i];
    }
    return sum;
}

#endif
