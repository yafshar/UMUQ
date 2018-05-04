#ifndef UMHBM_DCPSE_H
#define UMHBM_DCPSE_H

#include "polynomial.hpp"
#include "factorial.hpp"
#include "eigenmatrix.hpp"
#include "knearestneighbors.hpp"

/*! \class primitive
  * \brief Primitive function
  * 
  * \tparam T   data type
  * \tparan TF function type
  */
template <typename T, class TF>
class primitive
{
  public:
    inline T f(T const *x)
    {
        return static_cast<TF *>(this)->f(x);
    }

  private:
    friend TF;
};

/*!
 * \brief Primitive function (quartic spline)
 * 
 * Reference: Chen et al., Int. J. Numer. Meth. Engng 2003; 56:935–960.
 * 
 * \returns \f$ 1 - 6 x^2 + 8 x^3 - 3 x^4 \f$
 */
template <typename T>
class quartic_spline : public primitive<T, quartic_spline<T>>
{
  public:
    inline T f(T const *x)
    {
        return (*x > static_cast<T>(1)) ? T{} : 1 + (*x) * (*x) * (-6 + (*x) * (8 - 3 * (*x)));
    }
};

/*! \class dcpse
 * \brief 
 * 
 */
template <typename T>
class dcpse
{
  public:
    /*!
     * \brief constructor
     * 
     * \param ndim             Dimensiononality
     */
    dcpse(int ndim) : nDim(ndim) {}

    /*!
     * \brief 
     * 
     * \param idata            A pointer to input data 
     * \param nPoints          Number of data points
     * \param minDist          Local characteristic inter-points spacing
     * \param nearestneighbors Nearest neighbours for each point
     * \param nNN              Number of nearest neighbors
     * \param kernel           Operator kernel
     * \param beta             In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right) \f$
     *                         Notation for partial derivatives:
     *                         \f$ \begin{align} D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \label{eq:1} \end{align} \f$
     * \param order            Order of accuracy (default is 2nd order accurate)
     * \param ratio            The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool dcops(T *idata, int const nPoints, T *minDist, int *NearestNeighbors, int nNN, T *kernel, int *beta, int order = 2, T ratio = static_cast<T>(1))
    {
        // \f$ |\beta| = \beta_1 + \cdots + \beta_d \f$
        int Beta = std::accumulate(beta, beta + nDim, 0);
        if (Beta == 0)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Zero degree derivative is an interpolation!" << e.what() << std::endl;
            std::cerr << "Use the interpolation function!" << e.what() << std::endl;
            return false;
        }

        T rhscoeff = (Beta % 2 == 0) ? static_cast<T>(1) : -static_cast<T>(1);

        //Create an instance of polynomial object
        polynomial<T> poly(nDim, order);

        //size of the monomials
        int msize = poly.monomialsize();

        //get the monomial basis
        int *alpha = poly.monomial_basis();

        EVectorX<T> B0;
        B0.resize(msize);

        for (int i = 0, id = 0; i < msize; i++)
        {
            int maxalpha = 0;
            for (int j = 0; j < nDim; j++, id++)
            {
                maxalpha = std::max(maxalpha, std::abs(alpha[id] - beta[j]));
            }
            if (maxalpha)
            {
                B0(i) = T{};
            }
            else
            {
                T fact = static_cast<T>(1);
                std::for_each(beta, beta + nDim, [&](int const b_j) { fact *= factorial<T>(b_j); });
                B0(i) = rhscoeff * fact;
            }
        }
    }

  private:
    //Dimensiononality
    int nDim;
};

#endif
