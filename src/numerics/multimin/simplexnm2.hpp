#ifndef UMUQ_SIMPLEXNM2_H
#define UMUQ_SIMPLEXNM2_H

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

/*! \class simplexNM2
 * \ingroup Multimin_Module
 *
 * \brief The Simplex method of Nelder and Mead, also known as the polytope search algorithm.
 * It uses fixed coordinate axes around the starting point x to initialize the simplex.
 * The size of simplex is calculated as the RMS distance of each vertex from the center.
 *
 * \tparam DataType Data type
 *
 * Reference: <br>
 * Nelder, J.A., Mead, R., Computer Journal 7 (1965) pp. 308-313.
 *
 * This implementation uses \f$ n+1 \f$ corner points in the simplex.
 */
template <typename DataType>
class simplexNM2 : public functionMinimizer<DataType>
{
  public:
    /*!
     * \brief Construct a new simplexNM object
     *
     * \param Name Minimizer name
     */
    explicit simplexNM2(const char *Name = "simplexNM2");

    /*!
     * \brief Destroy the simplexNM object
     *
     */
    ~simplexNM2();

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
     * \returns false If it encounters an unexpected problem
     */
    bool init();

    /*!
     * \brief Drives the iteration of each algorithm
     *
     * It performs one iteration to update the state of the minimizer.
     *
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
     * \returns Function value at X
     */
    DataType tryCornerMove(DataType const coeff, int const corner, std::vector<DataType> &X);

    /*!
     * \brief Update the point
     *
     * \param i    Point index
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
     * \param best   The best corner
     * \param X      Input point
     *
     * \returns false If it encounters an unexpected problem
     */
    bool contractByBest(int const best, std::vector<DataType> &X);

    /*!
     * \brief Calculates the center of the simplex
     *
     * \returns false If it encounters an unexpected problem
     */
    bool computeCenter();

    /*!
     * \brief Compute the specific characteristic size
     *
     * The size of simplex is calculated as the RMS distance of each vertex from the center rather than
     * the mean distance, allowing a linear update of this quantity on each step.
     *
     * \returns DataType Computed characteristic size
     */
    DataType computeSize();

  private:
    /*!
     * \returns a (pointer to a) row of the data.
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

    //! Simplex characteristic size
    DataType S2;
};

template <typename DataType>
simplexNM2<DataType>::simplexNM2(const char *Name) : functionMinimizer<DataType>(Name), S2(DataType{}) {}

template <typename DataType>
simplexNM2<DataType>::~simplexNM2() {}

template <typename DataType>
bool simplexNM2<DataType>::reset(int const nDim) noexcept
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

    return true;
}

template <typename DataType>
bool simplexNM2<DataType>::init()
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
        // Copy the elements of the x to xtemp
        std::copy(this->x.begin(), this->x.end(), this->ws1.begin());

        // Currently ws2 contains stepSize from set
        this->ws1[i] += this->ws2[i];

        this->fval = this->fun.f(this->ws1.data());

        if (!std::isfinite(this->fval))
        {
            UMUQFAILRETURN("Non-finite function value encountered!");
        }

        // Copy the elements of the vector xtemp into the (i+1)-th row of the matrix x1
        std::ptrdiff_t const Id = (i + 1) * n;

        std::copy(this->ws1.begin(), this->ws1.end(), x1.data() + Id);

        y1[i + 1] = this->fval;
    }

    computeCenter();

    // Initialize simplex size
    this->characteristicSize = computeSize();

    return true;
}

template <typename DataType>
bool simplexNM2<DataType>::iterate()
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
DataType simplexNM2<DataType>::tryCornerMove(DataType const coeff, int const corner, std::vector<DataType> &X)
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
void simplexNM2<DataType>::updatePoint(int const i, std::vector<DataType> const &X, DataType const val)
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
bool simplexNM2<DataType>::contractByBest(int const best, std::vector<DataType> &X)
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
bool simplexNM2<DataType>::computeCenter()
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
DataType simplexNM2<DataType>::computeSize()
{
    int const n = this->getDimension();

    // Calculates simplex size as rms sum of length of vectors
    // from simplex center to corner points:
    // \f$ sqrt( sum ( || y - y_middlepoint ||^2 ) / n ) \f$
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
inline DataType *simplexNM2<DataType>::operator[](std::size_t const index) const
{
    int const n = this->getDimension();
    return x1.data() + index * n;
}

} // namespace multimin
} // namespace umuq

#endif // UMUQ_SIMPLEXNM2
