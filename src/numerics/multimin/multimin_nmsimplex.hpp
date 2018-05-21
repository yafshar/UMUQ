#ifndef UMUQ_MULTIMIN_NSIMPLEX_H
#define UMUQ_MULTIMIN_NSIMPLEX_H

/*! \class nmsimplex
 *  \ingroup multimin_Module
 * 
 * \brief 
 * 
 * The Simplex method of Nelder and Mead,
 * also known as the polytope search alogorithm. Ref:
 * Nelder, J.A., Mead, R., Computer Journal 7 (1965) pp. 308-313.
 * 
 * This implementation uses n+1 corner points in the simplex.
 * 
 * \tparam T   data type
 * \tparam TMF multimin function type
 */
template <typename T, class TMF>
class nmsimplex : public multimin_fminimizer_type<T, nmsimplex<T, TMF>, TMF>
{
  public:
    /*!
     * \brief constructor
     * 
     * \param name name of the differentiable function minimizer type (default "nmsimplex")
     */
    nmsimplex(const char *name_ = "nmsimplex") : x1(nullptr),
                                                 y1(nullptr),
                                                 ws1(nullptr),
                                                 ws2(nullptr) { this->name = name_; }

    /*!
     * \brief destructor
     */
    ~nmsimplex() { free(); }

    /*!
     * \brief allocate space for data type T
     * 
     * \param n_ size of array
     * 
     * \returns false if there is insufficient memory to create data array 
     */
    bool alloc(std::size_t const n_)
    {
        if (n_ <= 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Invalid number of parameters specified : " << std::endl;
            return false;
        }

        n = n_;
        try
        {
            std::ptrdiff_t const N = (n + 1) * n;
            x1 = new T[N];
            y1 = new T[n + 1];
            ws1 = new T[n];
            ws2 = new T[n];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    /*!
     * \brief set
     * 
     */
    bool set(TMF *f, T const *x, T *size, T const *step_size)
    {
        T *xtemp = ws1;

        //First point is the original x0
        T val = f->f(x);

        if (!std::isfinite(val))
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Non-finite function value encountered ! " << std::endl;
            return false;
        }

        //Copy the elements of the vector x into the 0-th row of the matrix x1
        std::copy(x, x + n, x1);

        y1[0] = val;

        //Following points are initialized to x0 + step_size
        for (std::size_t i = 0; i < n; i++)
        {
            //Copy the elements of the x to xtemp
            std::copy(x, x + n, xtemp);

            val = xtemp[i] + step_size[i];

            xtemp[i] = val;

            val = f->f(xtemp);

            if (!std::isfinite(val))
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << "Non-finite function value encountered ! " << std::endl;
                return false;
            }

            //Copy the elements of the vector xtemp into the (i+1)-th row of the matrix x1
            std::ptrdiff_t const Id = (i + 1) * n;
            std::copy(xtemp, xtemp + n, x1 + Id);

            y1[i + 1] = val;
        }

        //Initialize simplex size
        *size = compute_size();

        return true;
    }

    void free()
    {
        if (x1 != nullptr)
        {
            delete[] x1;
            x1 = nullptr;
        }
        if (y1 != nullptr)
        {
            delete[] y1;
            y1 = nullptr;
        }
        if (ws1 != nullptr)
        {
            delete[] ws1;
            ws1 = nullptr;
        }
        if (ws2 != nullptr)
        {
            delete[] ws2;
            ws2 = nullptr;
        }
    }

    bool iterate(TMF *f, T *x, T *size, T *fval)
    {
        //Simplex iteration tries to minimize function f value
        //xc and xc2 vectors store tried corner point coordinates
        T *xc = ws1;
        T *xc2 = ws2;

        std::size_t hi;
        std::size_t s_hi;
        std::size_t lo;

        T dhi;
        T dlo;
        T ds_hi;

        T val;
        T val2;

        //Get index of highest, second highest and lowest point
        dhi = dlo = y1[0];

        hi = 0;
        lo = 0;

        ds_hi = y1[1];
        s_hi = static_cast<std::size_t>(1);

        for (std::size_t i = 1; i < n; i++)
        {
            val = y1[i];
            if (val < dlo)
            {
                dlo = val;
                lo = i;
            }
            else if (val > dhi)
            {
                ds_hi = dhi;
                s_hi = hi;
                dhi = val;
                hi = i;
            }
            else if (val > ds_hi)
            {
                ds_hi = val;
                s_hi = i;
            }
        }

        //Reflect the highest value
        val = move_corner(static_cast<T>(-1), hi, xc, f);

        if (std::isfinite(val) && val < y1[lo])
        {

            //Reflected point becomes lowest point, try expansion
            val2 = move_corner(static_cast<T>(-2), hi, xc2, f);

            if (std::isfinite(val2) && val2 < y1[lo])
            {
                //Copy the elements of the vector xc2 into the hi-th row of the matrix x1
                std::ptrdiff_t const Id = hi * n;
                std::copy(xc2, xc2 + n, x1 + Id);

                y1[hi] = val2;
            }
            else
            {
                //Copy the elements of the vector xc into the hi-th row of the matrix x1
                std::ptrdiff_t const Id = hi * n;
                std::copy(xc, xc + n, x1 + Id);

                y1[hi] = val;
            }
        }
        //Reflection does not improve things enough
        //or
        //we got a non-finite (illegal) function value
        else if (!std::isfinite(val) || val > y1[s_hi])
        {
            if (std::isfinite(val) && val <= y1[hi])
            {

                //If trial point is better than highest point, replace highest point
                //Copy the elements of the vector xc into the hi-th row of the matrix x1
                std::ptrdiff_t const Id = hi * n;
                std::copy(xc, xc + n, x1 + Id);

                y1[hi] = val;
            }

            //Try one dimensional contraction
            val2 = move_corner(static_cast<T>(0.5), hi, xc2, f);

            if (std::isfinite(val2) && val2 <= y1[hi])
            {
                //Copy the elements of the vector xc2 into the hi-th row of the matrix x1
                std::ptrdiff_t const Id = hi * n;
                std::copy(xc2, xc2 + n, x1 + Id);

                y1[hi] = val2;
            }
            else
            {
                //Contract the whole simplex in respect to the best point
                if (!contract_by_best(lo, xc, f))
                {
                    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                    std::cerr << "contract_by_best failed ! " << std::endl;
                    return false;
                }
            }
        }
        else
        {
            //Trial point is better than second highest point. Replace highest point by it
            //Copy the elements of the vector xc into the hi-th row of the matrix x1
            std::ptrdiff_t const Id = hi * n;
            std::copy(xc, xc + n, x1 + Id);

            y1[hi] = val;
        }

        //Return lowest point of simplex as x
        lo = (std::size_t)std::distance(y1, std::min_element(y1, y1 + n));

        //Copy the elements of the lo-th row of the matrix x1 into the vector x
        {
            std::ptrdiff_t const Id = lo * n;
            std::copy(x1 + Id, x1 + Id + n, x);
        }

        *fval = y1[lo];

        //Update simplex size
        *size = compute_size();

        return true;
    }

    T move_corner(T const coeff, std::size_t corner, T *xc, TMF *f)
    {
        //Moves a simplex corner scaled by coeff (negative value represents
        //mirroring by the middle point of the "other" corner points)
        //and gives new corner in xc and function value at xc as a
        //return value
        T newval;
        T mp;

        //We have a in row-major order matrix \f$ (n+1) \times n \f$
        for (std::size_t j = 0; j < n; j++)
        {
            mp = T{};
            for (std::size_t i = 0; i < n + 1; i++)
            {
                if (i != corner)
                {
                    std::ptrdiff_t const Id = i * n + j;

                    mp += x1[Id];
                }
            }

            mp /= static_cast<T>(n);

            std::ptrdiff_t const Idx = corner * n + j;

            newval = mp - coeff * (mp - x1[Idx]);

            xc[j] = newval;
        }

        newval = f->f(xc);

        return newval;
    }

    bool contract_by_best(std::size_t best, T *xc, TMF *f)
    {
        //Function contracts the simplex in respect to
        //best valued corner. That is, all corners besides the
        //best corner are moved.

        //The xc vector is simply work space here
        T newval;

        for (std::size_t i = 0; i < n + 1; i++)
        {
            if (i != best)
            {
                std::ptrdiff_t Id = i * n;
                std::ptrdiff_t Idb = best * n;

                for (std::size_t j = 0; j < n; j++, Id++, Idb++)
                {
                    newval = static_cast<T>(0.5) * (x1[Id] + x1[Idb]);
                    x1[Id] = newval;
                }

                //Evaluate function in the new point
                Id = i * n;

                //Copy the elements of the i-th row of the matrix x1 into the vector xc
                std::copy(x1 + Id, x1 + Id + n, xc);

                newval = f->f(xc);

                y1[i] = newval;

                //Notify caller that we found at least one bad function value.
                //we finish the contraction (and do not abort) to allow the user
                //to handle the situation
                if (!std::isfinite(newval))
                {
                    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                    std::cerr << " The iteration encountered a singular point where the function or its derivative evaluated to Inf or NaN.! " << std::endl;
                    return false;
                }
            }
        }

        return true;
    }

    bool compute_center(T *mp)
    {
        //Calculates the center of the simplex to mp
        T val;
        for (std::size_t j = 0; j < n; j++)
        {
            val = T{};
            for (std::size_t i = 0; i < n + 1; i++)
            {
                std::ptrdiff_t const Id = i * n + j;
                val += x1[Id];
            }

            val /= static_cast<T>(n + 1);

            mp[j] = val;
        }

        return true;
    }

    T compute_size()
    {
        //Calculates simplex size as average sum of length of vectors
        //from simplex center to corner points:
        //\f$ (sum(|| y - y_middlepoint ||)) / n * \f$

        T *s = ws1;
        T *mp = ws2;

        std::size_t const N = n + 1;

        //Calculate middle point
        compute_center(mp);

        T ss(0);
        for (std::size_t i = 0; i < N; i++)
        {
            std::ptrdiff_t const Id = i * n;

            //Copy the elements of the i-th row of the matrix x1 into the vector s
            std::copy(x1 + Id, x1 + Id + n, s);

            for (std::size_t j = 0; j < n; j++)
            {
                s[j] -= mp[j];
            }

            {
                //Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
                T sum(0);
                std::for_each(s, s + n, [&](T const s_i) { sum += s_i * s_i; });

                ss += std::sqrt(sum);
            }
        }

        return ss / static_cast<T>(N);
    }

  private:
    //Simplex corner points (Matrix of size \f$ (n+1) \times n \f$
    T *x1;

    //Function value at corner points with size \f$ (n+1) \f$
    T *y1;

    //Workspace 1 for algorithm
    T *ws1;

    //Workspace 2 for algorithm
    T *ws2;

    std::size_t n;
};

#endif
