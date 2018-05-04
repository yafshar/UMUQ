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
     * \brief constructor
     * 
     * \param dm  Dimension
     * \param ord Order (the default order or degree of r in a space of dm dimensions is 2)
     */
    polynomial(unsigned int dim, unsigned int ord = 2) : nDim(dim), Order(ord)
    {
        if (nDim <= 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Fatal error! dimension <= 0!" << std::endl;
            throw(std::runtime_error("Can not have dimension <= 0!"));
        }
        if (Order < 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Fatal error! maximum accuracy order < 0!" << std::endl;
            throw(std::runtime_error("Maximum accuracy order < 0!"));
        }

        monomialSize = binomial_coefficient(nDim + Order, Order);
    }

    /*! 
     * \brief reset
     * 
     * Reset the values to the new ones
     * 
     * \param dm  new Dimension
     * \param ord new Order (the default order or degree of r in a space of dm dimensions is 2)
     */
    void reset(unsigned int dim, unsigned int ord = 2)
    {
        nDim = dim;
        if (nDim <= 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Fatal error! dimension <= 0!" << std::endl;
            throw(std::runtime_error("Can not have dimension <= 0!"));
        }

        Order = ord;
        if (Order < 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Fatal error! maximum accuracy order < 0!" << std::endl;
            throw(std::runtime_error("Maximum accuracy order < 0!"));
        }

        monomialSize = binomial_coefficient(nDim + Order, Order);

        alpha.reset(nullptr);
    }

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
    int binomial_coefficient(int const n, int const k)
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
     * \brief  \f$ \alpha = \f$ all the monomials in a d dimensional space, with total degree r.
     *   
     * For example:
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
     *       monomial_basis_(d=2,r=2)   = {1,    x,   y,  x^2, xy,  y^2}
     *                            alpha = {0,0, 1,0, 0,1, 2,0, 1,1, 0,2}
     *
     *
     *       monomial_basis_(d=3,r=2)   = {1,       x,     y,     z,    x^2,  xy,    xz,   y^2,    yz,    z^2  }
     *                            alpha = {0,0,0, 1,0,0, 0,1,0, 0,0,1, 2,0,0 1,1,0, 1,0,1, 0,2,0, 0,1,1, 0,0,2 }
     *
     *
     * \returns A pointer to monomial sequence
     */
    int *monomial_basis()
    {
        if (alpha)
        {
            return alpha.get();
        }
        else
        {
            int const N = nDim * monomialSize;
            alpha.reset(new int[N]);

            int x[nDim];
            std::fill(x, x + nDim, 0);

            int n = 0;

            for (;;)
            {
                for (int j = nDim - 1; j >= 0; j--, n++)
                {
                    alpha[n] = x[j];
                }

                if (x[0] == Order)
                {
                    return alpha.get();
                }

                if (!graded_reverse_lexicographic_order(x))
                {
                    return nullptr;
                }
            }
            return alpha.get();
        }
    };

    /*! 
     * \brief Evaluates a monomial at a point x.
     * 
     * \param  x       The coordinates of the evaluation points
     * \param  value   Monomial_value, the array value of the monomial at point x
     * 
     * \returns the size of the monomial array
     */
    int monomial_value(T *x, T *&value)
    {
        if (!alpha)
        {
            //Have to create monomial sequence
            int *tmp = monomial_basis();

            if (tmp == nullptr)
            {
                return 0;
            }
        }

        if (value == nullptr)
        {
            try
            {
                value = new T[monomialSize];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return 0;
            }
        }

        for (int i = 0, k = 0; i < monomialSize; i++)
        {
            T v = static_cast<T>(1);
            for (int j = 0; j < nDim; j++, k++)
            {
                v *= std::pow(x[j], alpha[k]);
            }
            value[i] = v;
        }

        return monomialSize;
    };

    int const monomialsize() const
    {
        return monomialSize;
    }

    int const dim() const
    {
        return nDim;
    }

    int const order() const
    {
        return Order;
    }

  private:
    /*! 
     * \brief Use a reverse lexicographic order for next monomial, degrees between 0 and r
     *  all monomials in a d dimensional space, with order of accuracy r.
     *
     * \param x   Current monomial
     * \param x   Next monomial, last value in the sequence is r.
     */
    bool graded_reverse_lexicographic_order(int *x)
    {
        if (Order == 0)
        {
            return true;
        }

        int asum = std::accumulate(x, x + nDim, 0);

        if (asum < 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Fatal error! input sums < 0" << std::endl;
            return false;
        }

        if (Order < asum)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Fatal error! input sums > maximum degree r" << std::endl;
            return false;
        }

        if (x[0] == Order)
        {
            x[0] = 0;
            x[nDim - 1] = 0;
        }
        else
        {
            int i;
            int tmp;

            //Seeking the first index in which x > 0.
            int j = 0;
            for (i = 1; i < nDim; i++)
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
                x[nDim - 1] = tmp + 1;
            }
            else if (j < nDim - 1)
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

  private:
    //Dimension
    int nDim;
    //Order of accuracy
    int Order;

    //The size of the monomial array
    int monomialSize;

    std::unique_ptr<int[]> alpha;
};

#endif
