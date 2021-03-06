#ifndef UMUQ_SIMPLEXNM2RND_H
#define UMUQ_SIMPLEXNM2RND_H

#include "core/core.hpp"
#include "numerics/function/functionminimizer.hpp"

#include <cmath>
#include <cstddef>

#include <vector>
#include <algorithm>
#include <iterator>

namespace umuq
{

inline namespace multimin
{

/*! \class simplexNM2Rnd
 * \ingroup Multimin_Module
 *
 * \brief The Simplex method of Nelder and Mead, also known as the polytope search algorithm.
 * It uses a randomly-oriented set of basis vectors instead of the fixed coordinate axes
 * around the starting point x to initialize the simplex.
 *
 * \tparam DataType Data type
 *
 * Reference:<br>
 * Nelder, J.A., Mead, R., Computer Journal 7 (1965) pp. 308-313.
 *
 * This implementation uses \f$ n+1 \f$ corner points in the simplex.
 */
template <typename DataType>
class simplexNM2Rnd : public functionMinimizer<DataType>
{
public:
    /*!
     * \brief Construct a new simplex N M2 Rnd object
     *
     * \param Name Minimizer name
     */
    explicit simplexNM2Rnd(const char *Name = "simplexNM2Rnd");

    /*!
     * \brief Destroy the simplexNM2Rnd object
     *
     */
    ~simplexNM2Rnd();

    /*!
     * \brief Resizes the minimizer vectors to contain nDim elements
     *
     * \param nDim  New size of the minimizer vectors
     *
     * \returns false If it encounters an unexpected problem
     */
    bool reset(int const nDim) noexcept;

    /*!
     * \brief Initialize the minimizer
     *
     * It uses a randomly-oriented set of basis vectors instead of the fixed coordinate axes
     * around the starting point x to initialize the simplex.<br>
     * The final dimensions of the simplex are scaled along the coordinate axes by the
     * vector step_size. <br>
     * The randomization uses a simple deterministic generator so that repeated calls to
     * functionMinimizer set for a given solver object will vary the orientation in a
     * well-defined way.
     *
     * Reference:<br>
     * https://www.gnu.org/software/gsl/doc/html/multimin.html
     *
     * \returns false If it encounters an unexpected problem
     */
    bool init();

    /*!
     * \brief Drives the iteration of each algorithm
     *
     * It performs one iteration to update the state of the minimizer.
     *
     * \returns true
     * \returns false If the iteration encounters an unexpected problem
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
     * \returns DataType Function value at X
     */
    DataType tryCornerMove(DataType const coeff, int const corner, std::vector<DataType> &X);

    /*!
     * \brief
     *
     * \param i    Index number
     * \param X    Input point
     * \param val
     */
    void updatePoint(int const i, std::vector<DataType> const &X, DataType const val);

    /*!
     * \brief Function contracts the simplex in respect to best valued corner.
     *
     * All corners besides the best corner are moved. The X vector is simply work space here
     * (This function is rarely called in practice, since it is the last choice, hence not optimized)
     *
     *
     * \param best   Best corner
     * \param X      Input point
     *
     * \returns true
     * \returns false If the iteration encounters an unexpected problem
     */
    bool contractByBest(int const best, std::vector<DataType> &X);

    /*!
     * \brief Compute the center of the simplex
     *
     * \returns false If the iteration encounters an unexpected problem
     */
    bool computeCenter();

    /*!
     * \brief Compute the specific characteristic size
     *
     * The size of simplex is calculated as the RMS distance of each vertex from the center rather than
     * the mean distance, allowing a linear update of this quantity on each step.
     *
     * \returns Computed characteristic size
     */
    DataType computeSize();

private:
    /*!
     * \class submatrix
     *
     * Returns memory id of an element in a matrix view of a submatrix of the matrix x1.
     * The upper-left element of the submatrix is the element (k1,k2) of the original
     * matrix. The submatrix has n1 rows and n2 columns.
     * The physical number of columns in memory given by NC is unchanged.
     * Mathematically, the (i,j)-th element of the new matrix is given by,<br>
     * \f$ ID(i, j)_{(NC, k1, k2, n1, n2)} = [(k1 * NC + k2) + i*NC + j ] \f$
     *
     */
    class submatrix
    {
    public:
        /*!
         * \brief Construct a new submatrix object
         *
         * \param NR  number of rows in the original matrix
         * \param NC_ number of columns in the original matrix
         * \param k1_ row number of the upper-left element of the submatrix
         * \param k2_ column number of the upper-left element of the submatrix
         * \param n1_ submatrix number of rows
         * \param n2_ submatrix number of columns
         */
        submatrix(int NR, int NC_, int k1_, int k2_, int n1_, int n2_);

        /*!
         * \brief memory ID of an element in a matrix view of a submatrix of the matrix x1
         *
         * \Returns memory id of an element in a matrix view of a submatrix of the matrix x1
         */
        inline std::ptrdiff_t ID(int i, int j) const;

    private:
        int NC;
        int k1;
        int k2;
        int n1;
        int n2;
    };

private:
    /*!
     * \brief Uniform RNG
     *
     * \param seed Seed to initialize the PRNG
     *
     * \returns DataType Uniform random number
     */
    inline DataType ran_unif(unsigned long *seed);

    /*!
     * \returns A (pointer to a) row of the data.
     */
    inline DataType *operator[](std::size_t const index) const;

private:
    //! Simplex corner points (Matrix of size \f$ (n+1) \times n \f$
    std::vector<DataType> x1;

    //! Function value at corner points with size \f$ (n+1) \f$
    std::vector<DataType> y1;

    //! Center of all points
    std::vector<DataType> center;

    //! Current step
    std::vector<DataType> delta;

    //! x - center (workspace)
    std::vector<DataType> xmc;

    //! Stored squared size
    DataType S2;

    //! counter
    unsigned long count;
};

template <typename DataType>
simplexNM2Rnd<DataType>::simplexNM2Rnd(const char *Name) : functionMinimizer<DataType>(Name) {}

template <typename DataType>
simplexNM2Rnd<DataType>::~simplexNM2Rnd() {}

template <typename DataType>
bool simplexNM2Rnd<DataType>::reset(int const nDim) noexcept
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

    center.resize(nDim);
    delta.resize(nDim);
    xmc.resize(nDim);

    count = 0;

    return true;
}

template <typename DataType>
bool simplexNM2Rnd<DataType>::init()
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

    {
        submatrix m(n + 1, n, 1, 0, n, n);

        // Set the elements of the submatrix m to the corresponding elements of the identity matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::ptrdiff_t const Id = m.ID(i, j);

                x1[Id] = (i == j) ? static_cast<DataType>(1) : DataType{};
            }
        }

        // Generate a random orthonormal basis
        unsigned long seed = count ^ 0x12345678;

        // Warm it up
        ran_unif(&seed);

        // Start with random reflections
        for (int i = 0; i < n; i++)
        {
            DataType s = ran_unif(&seed);
            if (s > 0.5)
            {
                std::ptrdiff_t const Id = m.ID(i, i);

                x1[Id] = -static_cast<DataType>(1);
            }
        }

        // Apply random rotations
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                // Rotate columns i and j by a random angle
                DataType const angle = M_2PI * ran_unif(&seed);
                DataType const c = std::cos(angle);
                DataType const s = std::sin(angle);

                // Apply a given rotation
                for (int r = 0; r < n; r++)
                {
                    std::ptrdiff_t const Id_ci = m.ID(r, i);
                    std::ptrdiff_t const Id_cj = m.ID(r, j);

                    DataType const x_r = x1[Id_ci];
                    DataType const y_r = x1[Id_cj];

                    x1[Id_ci] = c * x_r + s * y_r;
                    x1[Id_cj] = -s * x_r + c * y_r;
                }
            }
        }

        // Scale the orthonormal basis by the user-supplied step_size in
        // each dimension, and use as an offset from the central point x
        for (int i = 0; i < n; i++)
        {
            DataType const x_i = this->x[i];
            DataType const s_i = this->ws2[i];

            for (int j = 0; j < n; j++)
            {
                std::ptrdiff_t const Id_ij = m.ID(j, i);

                x1[Id_ij] *= s_i;
                x1[Id_ij] += x_i;
            }
        }

        // Compute the function values at each offset point
        for (int i = 0; i < n; i++)
        {
            std::ptrdiff_t const Id = m.ID(i, 0);

            DataType *r_i = x1.data() + Id;

            this->fval = this->fun.f(r_i);

            if (!std::isfinite(this->fval))
            {
                UMUQFAILRETURN("Non-finite function value encountered!");
            }

            y1[i + 1] = this->fval;
        }
    }

    computeCenter();

    // Initialize simplex size
    this->characteristicSize = computeSize();

    count++;

    return true;
}

template <typename DataType>
inline DataType simplexNM2Rnd<DataType>::ran_unif(unsigned long *seed)
{
    unsigned long s = *seed;
    *seed = (s * 69069 + 1) & 0xffffffffUL;
    return (*seed) / static_cast<DataType>(4294967296);
}

template <typename DataType>
bool simplexNM2Rnd<DataType>::iterate()
{
    // Simplex iteration tries to minimize function f value
    // ws1 and ws2 vectors store tried corner point coordinates

    int hi;
    int s_hi;
    int lo;

    DataType dhi;
    DataType dlo;
    DataType ds_hi;

    DataType val;
    DataType val2;

    // Get index of highest, second highest and lowest point
    dhi = dlo = y1[0];

    hi = 0;
    lo = 0;

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

    // Ty reflecting the highest value point
    val = tryCornerMove(-static_cast<DataType>(1), hi, this->ws1);

    if (std::isfinite(val) && val < y1[lo])
    {
        // Reflected point becomes lowest point, try expansion
        val2 = tryCornerMove(-static_cast<DataType>(2), hi, this->ws2);

        if (std::isfinite(val2) && val2 < y1[lo])
        {
            updatePoint(hi, this->ws2, val2);
        }
        else
        {
            updatePoint(hi, this->ws1, val);
        }
    }
    else if (!std::isfinite(val) || val > y1[s_hi])
    {
        // Reflection does not improve things enough, or we got a non-finite function value

        if (std::isfinite(val) && val <= y1[hi])
        {

            // If trial point is better than highest point, replace highest point
            updatePoint(hi, this->ws1, val);
        }

        // Try one dimensional contraction
        val2 = tryCornerMove(static_cast<DataType>(0.5), hi, this->ws2);

        if (std::isfinite(val2) && val2 <= y1[hi])
        {
            updatePoint(hi, this->ws2, val2);
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
        updatePoint(hi, this->ws1, val);
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
    // Recompute if accumulated error has made size invalid
    this->characteristicSize = (S2 > 0) ? std::sqrt(S2) : computeSize();

    return true;
}

template <typename DataType>
DataType simplexNM2Rnd<DataType>::tryCornerMove(DataType const coeff, int const corner, std::vector<DataType> &X)
{
    // \f$ N = n + 1 \f$
    // \f$ xc = (1-coeff)*((N)/(N-1)) * center(all) + ((N*coeff-1)/(N-1))*x_corner \f$

    int const n = this->getDimension();

    DataType const alpha = (1 - coeff) * (n + 1) / static_cast<DataType>(n);

    std::copy(center.begin(), center.end(), X.begin());

    std::for_each(X.begin(), X.end(), [&](DataType &x_i) { x_i *= alpha; });

    DataType const beta = ((n + 1) * coeff - 1) / static_cast<DataType>(n);

    std::ptrdiff_t const Id = corner * n;

    DataType *row = x1.data() + Id;

    for (int i = 0; i < n; i++)
    {
        X[i] += beta * row[i];
    }

    return this->fun.f(X.data());
}

template <typename DataType>
void simplexNM2Rnd<DataType>::updatePoint(int const i, std::vector<DataType> const &X, DataType const val)
{
    int const n = this->getDimension();

    std::ptrdiff_t const Id = i * n;

    DataType *x_orig = x1.data() + Id;

    // Compute \f$ delta = x - x_orig \f$
    std::copy(X.begin(), X.end(), delta.begin());

    for (int j = 0; j < n; j++)
    {
        delta[j] -= x_orig[j];
    }

    // Compute \f$ xmc = x_orig - c \f$
    std::copy(x_orig, x_orig + n, xmc.data());

    for (int j = 0; j < n; j++)
    {
        xmc[j] -= center[j];
    }

    DataType const N = static_cast<DataType>(n + 1);

    // Update size: \f$ S2' = S2 + (2/N) * (x_orig - c).delta + (N-1)*(delta/N)^2 \f$
    {
        DataType dsq(0);
        std::for_each(delta.begin(), delta.end(), [&](DataType const d_i) { dsq += d_i * d_i; });

        DataType s(0);
        for (int j = 0; j < n; j++)
        {
            s += xmc[j] * delta[j];
        }

        S2 += (static_cast<DataType>(2) / N) * s + (static_cast<DataType>(n) / N) * (dsq / N);
    }

    // Update center:  \f$ c' = c + (x - x_orig) / N \f$
    {
        DataType const alpha = static_cast<DataType>(1) / N;

        for (int j = 0; j < n; j++)
        {
            center[j] -= alpha * x_orig[j];
        }

        for (int j = 0; j < n; j++)
        {
            center[j] += alpha * X[j];
        }
    }

    // Copy the elements of the vector x into the i-th row of the matrix x1
    std::copy(X.begin(), X.end(), x_orig);

    y1[i] = val;
}

template <typename DataType>
bool simplexNM2Rnd<DataType>::contractByBest(int const best, std::vector<DataType> &X)
{
    std::ptrdiff_t const n = static_cast<std::ptrdiff_t>(this->getDimension());
    std::ptrdiff_t const b = static_cast<std::ptrdiff_t>(best);

    for (std::ptrdiff_t i = 0; i < n + 1; i++)
    {
        if (i != b)
        {
            DataType newval;

            std::ptrdiff_t Id = i * n;
            std::ptrdiff_t Idb = b * n;

            for (std::ptrdiff_t j = 0; j < n; j++, Id++, Idb++)
            {
                newval = static_cast<DataType>(0.5) * (x1[Id] + x1[Idb]);
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

    // We need to update the centre and size as well
    computeCenter();

    computeSize();

    return true;
}

template <typename DataType>
bool simplexNM2Rnd<DataType>::computeCenter()
{
    // Calculates the center of the simplex and stores in center
    std::fill(center.begin(), center.end(), DataType{});

    int const n = this->getDimension();

    for (int i = 0; i < n + 1; i++)
    {
        std::ptrdiff_t const Id = i * n;

        DataType *row = x1.data() + Id;

        for (int j = 0; j < n; j++)
        {
            center[j] += row[j];
        }
    }

    {
        DataType const alpha = static_cast<DataType>(1) / static_cast<DataType>(n + 1);

        std::for_each(center.begin(), center.end(), [&](DataType &c_i) { c_i *= alpha; });
    }

    return true;
}

template <typename DataType>
DataType simplexNM2Rnd<DataType>::computeSize()
{
    int const n = this->getDimension();

    // Calculates simplex size as rms sum of length of vectors
    // from simplex center to corner points:
    // \f$ \sqrt( \sum ( || y - y_middlepoint ||^2 ) / n ) \f$
    DataType s(0);
    for (int i = 0; i < n + 1; i++)
    {
        // Copy the elements of the i-th row of the matrix x1 into the vector s
        std::ptrdiff_t const Id = i * n;

        std::copy(x1.data() + Id, x1.data() + Id + n, this->ws1.begin());

        for (int j = 0; j < n; j++)
        {
            this->ws1[j] -= center[j];
        }

        DataType t(0);
        std::for_each(this->ws1.begin(), this->ws1.end(), [&](DataType const w_i) { t += w_i * w_i; });

        // squared size
        s += t;
    }

    // Store squared size
    S2 = s / static_cast<DataType>(n + 1);

    return std::sqrt(S2);
}

template <typename DataType>
inline DataType *simplexNM2Rnd<DataType>::operator[](std::size_t const index) const
{
    int const n = this->getDimension();
    return x1.data() + index * n;
}

template <typename DataType>
simplexNM2Rnd<DataType>::submatrix::submatrix(int NR, int NC_, int k1_, int k2_, int n1_, int n2_)
{
    if (k1_ > NR || k2_ > NC_ || n1_ > NR || n2_ > NC_)
    {
        UMUQMSG("Submatrix of size   : ", n1_, " x ", n2_);
        UMUQMSG("From matrix of size : ", NR, " x ", NC_);
        UMUQMSG("Start index of      : ", k1_, ", ", k2_);
        UMUQMSG("");
        UMUQFAIL("Input data overruns the end of the original matrix!");
    }
    NC = NC_;
    k1 = k1_;
    k2 = k2_;
    n1 = n1_;
    n2 = n2_;
}

template <typename DataType>
inline std::ptrdiff_t simplexNM2Rnd<DataType>::submatrix::ID(int i, int j) const
{
    return k1 * NC + k2 + i * NC + j;
}

} // namespace multimin
} // namespace umuq

#endif // UMUQ_SIMPLEXNM2RND
