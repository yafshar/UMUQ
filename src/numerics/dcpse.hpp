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
 * 
 * \tparam T    Data type
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
     * \param kernel           Operator kernel
     * \param beta             In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right) \f$
     *                         Notation for partial derivatives:
     *                         \f$ \begin{align} D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \label{eq:1} \end{align} \f$
     * \param order            Order of accuracy (default is 2nd order accurate)
	 * \param nENN             Number of extra nearest neighbors to aid in case of Vandermonde matrix sigularity (default is 2)
     * \param ratio            The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
	bool dcops(T *idata, int const nPoints, T *kernel, int *beta, int order = 2, int nENN = 2, T ratio = static_cast<T>(1))
	{
		// \f$ |\beta| = \beta_1 + \cdots + \beta_d \f$
		int Beta = std::accumulate(beta, beta + nDim, 0);
		if (Beta == 0)
		{
			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << "Zero order degree derivative is an interpolation!" << std::endl;
			std::cerr << "Use the interpolation function!" << std::endl;
			return false;
		}

		T rhscoeff = (Beta % 2 == 0) ? static_cast<T>(1) : -static_cast<T>(1);

		//Create an instance of polynomial object
		polynomial<T> poly(nDim, order);

		//size of the monomials
		int msize = poly.monomialsize();

		//Extra check on the number of extra nearest neighbors
		nENN = (nENN >= 0) ? nENN : 0;

		//Finding \f$ K = monomial size + 2 \f$ nearest neighbors
		L2NearestNeighbor<T> KNN(nPoints, nDim, msize + nENN);

		//Construct a kd-tree index & do nearest neighbors search
		KNN.buildIndex(idata);

		EVectorX<T> B0;
		B0.resize(msize);

		//Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
		//\f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
		{
			//get the monomial basis
			int *alpha = poly.monomial_basis();

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

			//TODO : check this again
			//When applicable, and for stability reasons, set the zeroth moment to 5
			if (rhscoeff > T{})
			{
				if (std::accumulate(alpha, alpha + nDim, 0) == 0)
				{
					B0(0) = static_cast<T>(5);
				}
			}
		}

		//Number of nearest neighbours for each point
		int nNN = KNN.numNearestNeighbors();

		//Array for keeping the component-wise L1 distances
		T L1Dist[nNN * nDim];

		//Loop over all points
		for (int i = 0; i < nPoints; i++)
		{
			int const IdI = i * nDim;

			{
				//A pointer to nearest neighbors indices of point i
				int *nearestneighbors = KNN.nearestneighbors(i);

				//A pointer to nearest neighbors distances from the point i
				T *nnDist = KNN.NearestNeighborsDistances(i);

				//A pointer to the array for keeping the component-wise L1 distances
				T *l1dist = L1Dist;

				for (int j = 0; j < nNN; j++)
				{
					int const IdJ = nearestneighbors[j] * nDim;

					T *Idata = idata + IdI;
					T *Jdata = idata + IdJ;

					for (int d = 0; d < nDim; d++)
					{
						*l1dist++ = *Jdata++ - *Idata++;
					}
				}
			}

			//Compute component-wise average neighbor spacing
			T sum(0);
			std::for_each(L1Dist, L1Dist + nNN * nDim, [&](T const l_i) { sum += std::abs(l_i); });

			//Component-wise average neighbor spacing
			sum /= static_cast<T>(nNN);

			//Computing the smoothing length for each point \frac{h}{\epsilon} \sim ratio \f$
			T const epsilon = sum / ratio;

			//Vectors pointing to x from all neighboring points
			std::for_each(L1Dist, L1Dist + nNN * nDim, [&](T &l_i) { l_i /= epsilon; });

			//Creating a transpose of the Vandermonde matrix
			//with the size of monomials * monomials
			EMatrixX<T> VT;
			VT.resize(msize, msize);

			for (int j = 0; j < msize; j++)
			{
				int const Id = j * nDim;

				//Evaluates a monomial at a point x
				T col[msize];
				poly.monomial_value(L1Dist + Id, col);

				VT.block(0, j, msize, 1) << col;
			}

			//LU decomposition of a matrix with complete pivoting, and related features.
			Eigen::FullPivLU<EMatrixX<T>> lu(VT);

			if (lu.rank() < msize)
			{
			}
			else
			{
			}
		}
	}

  private:
	//Dimensiononality
	int nDim;
};

#endif
