#ifndef UMUQ_SIMPLEXNM_H
#define UMUQ_SIMPLEXNM_H

namespace umuq
{

inline namespace multimin
{

/*! \class simplexNM
 * \ingroup Multimin_Module
 * 
 * \brief The Simplex method of Nelder and Mead, also known as the polytope search algorithm. 
 * It uses fixed coordinate axes around the starting point x to initialize the simplex.
 * The size of simplex is calculated as the mean distance of each vertex from the center.
 * 
 * Ref:
 * Nelder, J.A., Mead, R., Computer Journal 7 (1965) pp. 308-313.
 * 
 * This implementation uses n+1 corner points in the simplex.
 * 
 * \tparam T  Data type
 */
template <typename T>
class simplexNM : public functionMinimizer<T>
{
  public:
    /*!
     * \brief Construct a new simplexNM object
     * 
     * \param Name Minimizer name
     */
    explicit simplexNM(const char *Name = "simplexNM");

    /*!
     * \brief Destroy the simplexNM object
     * 
     */
    ~simplexNM();

    /*!
     * \brief Resizes the minimizer vectors to contain nDim elements
     *
     * \param nDim  New size of the minimizer vectors
     *
     * \returns true
     */
    bool reset(int const nDim) noexcept;

    /*!
     * \brief Initialize the minimizer
     * 
     * \return true 
     * \return false 
     */
    bool init();

    /*!
     * \brief Drives the iteration of each algorithm
     *
     * It performs one iteration to update the state of the minimizer.
     *
     * \return true
     * \return false If the iteration encounters an unexpected problem
     */
    bool iterate();

    /*!
     * \brief Moves a simplex corner scaled by coeff 
     * 
     * (negative value represents mirroring by the middle point of the "other" corner points)
     * and gives new corner in X and function value at X as a return value
     * 
     * \param coeff   Scaling coefficient
     * \param corner  Corner point
     * \param X       Input point
     * 
     * \return Function value at X
     */
    T moveCorner(T const coeff, int const corner, std::vector<T> &X);

    /*!
     * \brief Function contracts the simplex in respect to best valued corner.
     * 
     * All corners besides the best corner are moved. The X vector is simply work space here
     * 
     * \param best   best corner
     * \param X      Input point
     * 
     * \return true 
     * \return false 
     */
    bool contractByBest(int const best, std::vector<T> &X);

    /*!
     * \brief 
     * 
     * \param X 
     * \return true 
     * \return false 
     */
    bool computeCenter(std::vector<T> &X);

    /*!
     * \brief Compute the specific characteristic size
     *  
     * The size of simplex is calculated as the mean distance of each vertex from the center. 
     * 
     * \return Computed characteristic size
     */
    T computeSize();

  private:
    //! Simplex corner points (Matrix of size \f$ (n+1) \times n \f$
    std::vector<T> x1;

    //! Function value at corner points with size \f$ (n+1) \f$
    std::vector<T> y1;
};

template <typename T>
simplexNM<T>::simplexNM(const char *Name) : functionMinimizer<T>(Name) {}

template <typename T>
simplexNM<T>::~simplexNM() {}

template <typename T>
bool simplexNM<T>::reset(int const nDim) noexcept
{
    if (nDim <= 0)
    {
        UMUQFAILRETURN("Invalid number of parameters specified!");
    }

    this->x.resize(nDim);
    this->ws1.resize(nDim);
    this->ws2.resize(nDim);

    x1.resize((nDim + 1) * nDim);
    y1.resize(nDim + 1);

    return true;
}

template <typename T>
bool simplexNM<T>::init()
{
    int const n = this->getDimension();

    // First point is the original x0
    this->fval = this->fun.f(this->x.data());

    if (!std::isfinite(this->fval))
    {
        UMUQFAILRETURN("Non-finite function value encountered!");
    }

    // Copy the elements of the vector x into the 0-th row of the matrix x1
    std::copy(this->x.begin(), this->x.end(), x1.begin());

    y1[0] = this->fval;

    // Following points are initialized to x0 + step_size
    for (int i = 0; i < n; i++)
    {
        // Copy the elements of the x to ws1
        std::copy(this->x.begin(), this->x.end(), this->ws1.begin());

        // Currently ws2 contains stepSize from set
        this->ws1[i] += this->ws2[i];

        this->fval = this->fun.f(this->ws1.data());

        if (!std::isfinite(this->fval))
        {
            UMUQFAILRETURN("Non-finite function value encountered!");
        }

        // Copy the elements of the vector ws1 into the (i+1)-th row of the matrix x1
        std::ptrdiff_t const Id = (i + 1) * n;

        std::copy(this->ws1.begin(), this->ws1.end(), x1.data() + Id);

        y1[i + 1] = this->fval;
    }

    // Initialize simplex size
    this->characteristicSize = computeSize();

    return true;
}

template <typename T>
bool simplexNM<T>::iterate()
{
    // Simplex iteration tries to minimize function f value
    // ws1 and ws2 vectors store tried corner point coordinates

    int hi(0);
    int s_hi;
    int lo(0);

    T dhi;
    T dlo;
    T ds_hi;

    T val;
    T val2;

    // Get index of highest, second highest and lowest point
    dhi = dlo = y1[0];
    ds_hi = y1[1];
    s_hi = 1;

    int const n = this->getDimension();

    for (int i = 1; i < n + 1; i++)
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

    // Reflect the highest value
    val = moveCorner(-static_cast<T>(1), hi, this->ws1);

    if (std::isfinite(val) && val < y1[lo])
    {

        // Reflected point becomes lowest point, try expansion
        val2 = moveCorner(-static_cast<T>(2), hi, this->ws2);

        if (std::isfinite(val2) && val2 < y1[lo])
        {
            // Copy the elements of the vector ws2 into the hi-th row of the matrix x1
            std::ptrdiff_t const Id = hi * n;

            std::copy(this->ws2.begin(), this->ws2.end(), x1.data() + Id);

            y1[hi] = val2;
        }
        else
        {
            // Copy the elements of the vector ws1 into the hi-th row of the matrix x1
            std::ptrdiff_t const Id = hi * n;

            std::copy(this->ws1.begin(), this->ws1.end(), x1.data() + Id);

            y1[hi] = val;
        }
    }
    // Reflection does not improve things enough
    // or
    // we got a non-finite (illegal) function value
    else if (!std::isfinite(val) || val > y1[s_hi])
    {
        if (std::isfinite(val) && val <= y1[hi])
        {

            // If trial point is better than highest point, replace highest point
            // Copy the elements of the vector ws1 into the hi-th row of the matrix x1
            std::ptrdiff_t const Id = hi * n;

            std::copy(this->ws1.begin(), this->ws1.end(), x1.data() + Id);

            y1[hi] = val;
        }

        // Try one dimensional contraction
        val2 = moveCorner(static_cast<T>(0.5), hi, this->ws2);

        if (std::isfinite(val2) && val2 <= y1[hi])
        {
            // Copy the elements of the vector ws2 into the hi-th row of the matrix x1
            std::ptrdiff_t const Id = hi * n;

            std::copy(this->ws2.begin(), this->ws2.end(), x1.data() + Id);

            y1[hi] = val2;
        }
        else
        {
            // Contract the whole simplex in respect to the best point
            if (!contractByBest(lo, this->ws1))
            {
                UMUQFAILRETURN("contractByBest failed!");
            }
        }
    }
    else
    {
        // Trial point is better than second highest point. Replace highest point by it
        // Copy the elements of the vector ws1 into the hi-th row of the matrix x1
        std::ptrdiff_t const Id = hi * n;

        std::copy(this->ws1.begin(), this->ws1.end(), x1.data() + Id);

        y1[hi] = val;
    }

    // Return lowest point of simplex as x
    lo = static_cast<int>(std::distance(y1.begin(), std::min_element(y1.begin(), y1.end())));

    // Copy the elements of the lo-th row of the matrix x1 into the vector x
    {
        std::ptrdiff_t const Id = lo * n;

        std::copy(x1.data() + Id, x1.data() + Id + n, this->x.begin());
    }

    this->fval = y1[lo];

    // Update simplex size
    this->characteristicSize = computeSize();

    return true;
}

template <typename T>
T simplexNM<T>::moveCorner(T const coeff, int const corner, std::vector<T> &X)
{
    int const n = this->getDimension();

    // We have a row-major order matrix \f$ (n+1) \times n \f$
    for (int j = 0; j < n; j++)
    {
        T s(0);
        for (int i = 0; i < n + 1; i++)
        {
            if (i != corner)
            {
                std::ptrdiff_t const Id = i * n + j;

                s += x1[Id];
            }
        }

        s /= static_cast<T>(n);

        std::ptrdiff_t const Idx = corner * n + j;

        T newval = s - coeff * (s - x1[Idx]);

        X[j] = newval;
    }

    return this->fun.f(X.data());
}

template <typename T>
bool simplexNM<T>::contractByBest(int const best, std::vector<T> &X)
{
    std::ptrdiff_t const n = static_cast<std::ptrdiff_t>(this->getDimension());
    std::ptrdiff_t const b = static_cast<std::ptrdiff_t>(best);

    for (std::ptrdiff_t i = 0; i < n + 1; i++)
    {
        if (i != b)
        {
            T newval;

            std::ptrdiff_t Id = i * n;
            std::ptrdiff_t Idb = b * n;

            for (std::ptrdiff_t j = 0; j < n; j++, Id++, Idb++)
            {
                newval = static_cast<T>(0.5) * (x1[Id] + x1[Idb]);
                x1[Id] = newval;
            }

            // Evaluate function in the new point
            Id = i * n;

            // Copy the elements of the i-th row of the matrix x1 into the vector X
            std::copy(x1.data() + Id, x1.data() + Id + n, X.begin());

            newval = this->fun.f(X.data());

            y1[i] = newval;

            // Notify caller that we found at least one bad function value.
            // we finish the contraction (and do not abort) to allow the user
            // to handle the situation
            if (!std::isfinite(newval))
            {
                UMUQFAILRETURN("The iteration encountered a singular point where the function or its derivative evaluated to Inf or NaN!");
            }
        }
    }

    return true;
}

template <typename T>
bool simplexNM<T>::computeCenter(std::vector<T> &X)
{
    int const n = this->getDimension();

    // Calculates the center of the simplex to X
    for (int j = 0; j < n; j++)
    {
        T s(0);
        for (int i = 0; i < n + 1; i++)
        {
            std::ptrdiff_t const Id = i * n + j;
            s += x1[Id];
        }

        s /= static_cast<T>(n + 1);

        X[j] = s;
    }

    return true;
}

template <typename T>
T simplexNM<T>::computeSize()
{
    // Calculates simplex size as average sum of length of vectors
    // from simplex center to corner points:
    // \f$ (sum(|| y - y_middlepoint ||)) / n * \f$

    int const n = this->getDimension();

    // Calculate middle point
    computeCenter(this->ws2);

    T s(0);
    for (int i = 0; i < n + 1; i++)
    {
        std::ptrdiff_t const Id = i * n;

        // Copy the elements of the i-th row of the matrix x1 into the vector ws1
        std::copy(x1.data() + Id, x1.data() + Id + n, this->ws1.begin());

        for (int j = 0; j < n; j++)
        {
            this->ws1[j] -= this->ws2[j];
        }

        {
            //Compute the Euclidean norm \f$ ||x||_2 = \sqrt {\sum x_i^2} of the vector x = gradient. \f$
            T sum(0);
            std::for_each(this->ws1.begin(), this->ws1.end(), [&](T const w_i) { sum += w_i * w_i; });

            s += std::sqrt(sum);
        }
    }

    return s / static_cast<T>(n + 1);
}

} // namespace multimin
} // namespace umuq

#endif // UMUQ_SIMPLEXNM
