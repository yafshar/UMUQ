#ifndef UMHBM_DCPSE_H
#define UMHBM_DCPSE_H

#include "polynomial.hpp"
#include "factorial.hpp"
#include "eigenmatrix.hpp"
#include "knearestneighbors.hpp"
#include "primitive.hpp"

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
     * \brief Computes generalized DC operators 
     * 
     * \param idata            A pointer to input data 
     * \param nPoints          Number of data points
     * \param kernel           Operator kernel
     * 
     * \param beta             In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right) \f$
     *                         Notation for partial derivatives:
     *                         \f$ \begin{align} D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \label{eq:1} \end{align} \f$
     * \param order            Order of accuracy (default is 2nd order accurate)
	 * \param nENN             Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
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

        int alphamin = (Beta % 2 == 0);

        //\f$ (-1)^{|\beta|} \f$
        T rhscoeff = alphamin ? static_cast<T>(1) : -static_cast<T>(1);

        //Create an instance of polynomial object with polynomial degree of \f$ |\beta| + r -1 \f$
        polynomial<T> poly(nDim, order + Beta - 1);

        //Get the monomials size
        //\f$ msize = \left(\begin{matrix} |\beta| + r + d -1 \\ d \end{matrix}\right) - \alpha_{\min} \f$
        int msize = poly.monomialsize() - alphamin;

        if (kernel == nullptr)
        {
            try
            {
                kernel = new T[nPoints * msize];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }
        }

        //Extra check on the number of extra nearest neighbors
        nENN *= (nENN > 0) ? 1 : 0;

        //The number of points in the neighborhood of the operator
        //\f$ = \text{monomial size} + \text{number of extra neighbors} \f$ nearest neighbors
        //Finding nearest neighbors
        L2NearestNeighbor<T> KNN(nPoints, nDim, msize + nENN);

        //Construct a kd-tree index & do nearest neighbors search
        KNN.buildIndex(idata);

        //Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
        //\f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
        EVectorX<T> B0;
        B0.resize(msize);
        {
            //Get a pointer to the monomial basis
            int *alpha = poly.monomial_basis();

            for (int i = 0, id = alphamin * nDim; i < msize; i++)
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

        //Total number of nearest neighbours for each point
        int nNN = KNN.numNearestNeighbors();

        //Array for keeping the component-wise L1 distances
        T L1Dist[nNN * nDim];

        //Creating a transpose of the Vandermonde matrix
        //with the size of monomials * monomials \f$  = l \times l \f$
        EMatrixX<T> VMT;
        VMT.resize(msize, nNN);

        EVectorX<T> EM;
        EM.resize(nNN);

        EMatrixX<T> AM;
        AM.resize(msize, msize);

        EVectorX<T> SV;
        SV.resize(msize);

        //Loop over all points
        for (int i = 0; i < nPoints; i++)
        {
            std::ptrdiff_t const IdI = i * nDim;

            //For each point \f$ {\mathbf x} \f$ we define \f$ \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, \f$
            //as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
            {
                //A pointer to nearest neighbors indices of point i
                int *nearestneighbors = KNN.nearestneighbors(i);

                //A pointer to the array for keeping the component-wise L1 distances
                T *l1dist = L1Dist;

                //\f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = nearestneighbors[j] * nDim;

                    T *Idata = idata + IdI;
                    T *Jdata = idata + IdJ;

                    for (int d = 0; d < nDim; d++)
                    {
                        *l1dist++ = *Idata++ - *Jdata++;
                    }
                }
            }

            //Compute component-wise average neighbor spacing
            T h_avg(0);
            std::for_each(L1Dist, L1Dist + nNN * nDim, [&](T const l_i) { h_avg += std::abs(l_i); });

            //Component-wise average neighbor spacing \f$ h \f$
            h_avg /= static_cast<T>(nNN);

            //Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
            T const epsilon = h_avg / ratio;

            //Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
            std::for_each(L1Dist, L1Dist + nNN * nDim, [&](T &l_i) { l_i /= epsilon; });

            for (int j = 0; j < nNN; j++)
            {
                std::ptrdiff_t const Id = j * nDim;

                //Evaluates a monomial at a point \f$ {\mathbf x} \f$
                T col[msize + alphamin];
                poly.monomial_value(L1Dist + Id, col);

                VMT.block(0, j, msize, 1) << Eigen::Map<EVectorX<T>>(col + alphamin);
            }

            {
                T const epsilonsq = 2 * epsilon * epsilon;

                //A pointer to nearest neighbors distances from the point i
                T *nnDist = KNN.NearestNeighborsDistances(i);

                for (int j = 0; j < nNN; j++)
                {
                    EM(j) = std::exp(-nnDist[j] * nnDist[j] / epsilonsq);
                }
            }

            //\f$ \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
            //{\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
            //{\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
            //\end{matrix}
            EMatrixX<T> BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            //LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(BMT);

            //if necessary, remove redundant equations/coefficients
            if (lu.rank() < msize)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << "There are some singularities! we use a least-squares solution!" << std::endl;

                AM = BMT * BMT.transpose();

                Eigen::JacobiSVD<EMatrixX<T>> svd(AM);
                SV = svd.solve(B0);
            }
            else
            {
                EMatrixX<T> image = lu.image(BMT);
                AM = image * image.transpose();

                lu.compute(AM);

                SV = lu.solve(B0);
            }

        } //Loop over all points

        return true;
    }

    /*!
     * \brief Computes generalized DC operators 
     * 
     * \param idata            A pointer to input data 
     * \param nPoints          Number of data points
     * \param qdata            A pointer to query data 
     * \param nqPoints         Number of query data points
     * \param kernel           Operator kernel
     * 
     * \param beta             In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right) \f$
     *                         Notation for partial derivatives:
     *                         \f$ \begin{align} D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \label{eq:1} \end{align} \f$
     * \param order            Order of accuracy (default is 2nd order accurate)
	 * \param nENN             Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
     * \param ratio            The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool dcops(T *idata, int const nPoints, T *qdata, int const nqPoints, T *kernel, int *beta, int order = 2, int nENN = 2, T ratio = static_cast<T>(1))
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

        int alphamin = (Beta % 2 == 0);

        //\f$ (-1)^{|\beta|} \f$
        T rhscoeff = alphamin ? static_cast<T>(1) : -static_cast<T>(1);

        //Create an instance of polynomial object with polynomial degree of \f$ |\beta| + r -1 \f$
        polynomial<T> poly(nDim, order + Beta - 1);

        //Get the monomials size
        //\f$ msize = \left(\begin{matrix} |\beta| + r + d -1 \\ d \end{matrix}\right) - \alpha_{\min} \f$
        int msize = poly.monomialsize() - alphamin;

        if (kernel == nullptr)
        {
            try
            {
                kernel = new T[nqPoints * msize];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }
        }

        //Extra check on the number of extra nearest neighbors
        nENN = (nENN >= 0) ? nENN : 0;

        //The number of points in the neighborhood of the operator
        //\f$ = \text{monomial size} + \text{number of extra neighbors} \f$ nearest neighbors
        //Finding nearest neighbors
        L2NearestNeighbor<T> KNN(nPoints, nqPoints, nDim, msize + nENN);

        //Construct a kd-tree index & do nearest neighbors search
        KNN.buildIndex(idata, qdata);

        //Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
        //\f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
        EVectorX<T> B0;
        B0.resize(msize);
        {
            //Get a pointer to the monomial basis
            int *alpha = poly.monomial_basis();

            for (int i = 0, id = alphamin * nDim; i < msize; i++)
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

            //At off-particle locations it should be always zero to obtain kernels
            //with a vanishing zeroth-order moment that can be consistently evaluated
            B0(0) = T{};
        }

        //Total number of nearest neighbours for each point
        int nNN = KNN.numNearestNeighbors();

        //Array for keeping the component-wise L1 distances
        T L1Dist[nNN * nDim];

        //Creating a transpose of the Vandermonde matrix
        //with the size of monomials * monomials \f$  = l \times l \f$
        EMatrixX<T> VMT;
        VMT.resize(msize, nNN);

        EVectorX<T> EM;
        EM.resize(nNN);

        EMatrixX<T> AM;
        AM.resize(msize, msize);

        EVectorX<T> SV;
        SV.resize(msize);

        //Loop over all query points
        for (int i = 0; i < nqPoints; i++)
        {
            std::ptrdiff_t const IdI = i * nDim;

            //For each point \f$ {\mathbf x} \f$ we define \f$ \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, \f$
            //as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
            {
                //A pointer to nearest neighbors indices of point i
                int *nearestneighbors = KNN.nearestneighbors(i);

                //A pointer to the array for keeping the component-wise L1 distances
                T *l1dist = L1Dist;

                //\f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = nearestneighbors[j] * nDim;

                    //pointer to query data
                    T *Idata = qdata + IdI;
                    //pointer to idata
                    T *Jdata = idata + IdJ;

                    for (int d = 0; d < nDim; d++)
                    {
                        *l1dist++ = *Idata++ - *Jdata++;
                    }
                }
            }

            //Compute component-wise average neighbor spacing
            T h_avg(0);
            std::for_each(L1Dist, L1Dist + nNN * nDim, [&](T const l_i) { h_avg += std::abs(l_i); });

            //Component-wise average neighbor spacing \f$ h \f$
            h_avg /= static_cast<T>(nNN);

            //Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
            T const epsilon = h_avg / ratio;

            //Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
            std::for_each(L1Dist, L1Dist + nNN * nDim, [&](T &l_i) { l_i /= epsilon; });

            for (int j = 0; j < nNN; j++)
            {
                std::ptrdiff_t const Id = j * nDim;

                //Evaluates a monomial at a point \f$ {\mathbf x} \f$
                T col[msize + alphamin];
                poly.monomial_value(L1Dist + Id, col);

                VMT.block(0, j, msize, 1) << Eigen::Map<EVectorX<T>>(col + alphamin);
            }

            {
                T const epsilonsq = 2 * epsilon * epsilon;

                //A pointer to nearest neighbors distances from the point i
                T *nnDist = KNN.NearestNeighborsDistances(i);

                for (int j = 0; j < nNN; j++)
                {
                    EM(j) = std::exp(-nnDist[j] * nnDist[j] / epsilonsq);
                }
            }

            //\f$ \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
            //{\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
            //{\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
            //\end{matrix}
            EMatrixX<T> BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            //LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(BMT);

            //if necessary, remove redundant equations/coefficients
            if (lu.rank() < msize)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << "There are some singularities! we use a least-squares solution!" << std::endl;

                AM = BMT * BMT.transpose();

                Eigen::JacobiSVD<EMatrixX<T>> svd(AM);
                SV = svd.solve(B0);
            }
            else
            {
                EMatrixX<T> image = lu.image(BMT);
                AM = image * image.transpose();

                lu.compute(AM);

                SV = lu.solve(B0);
            }

        } //Loop over all points

        return true;
    }

    /*!
     * \brief Computes generalized interpolator DC operators 
     * 
     * \param idata            A pointer to input data 
     * \param nPoints          Number of data points
     * \param qdata            A pointer to query data 
     * \param nqPoints         Number of query data points
     * \param kernel           Operator kernel
     * \param order            Order of accuracy (default is 2nd order accurate)
	 * \param nENN             Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
     * \param ratio            The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool dcops(T *idata, int const nPoints, T *qdata, int const nqPoints, T *kernel, int order = 2, int nENN = 2, T ratio = static_cast<T>(1))
    {
        //Create an instance of polynomial object with polynomial degree of \f$ |\beta| + r -1 \f$
        polynomial<T> poly(nDim, order - 1);

        //Get the monomials size
        //\f$ msize = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$
        int msize = poly.monomialsize();

        if (kernel == nullptr)
        {
            try
            {
                kernel = new T[nqPoints * msize];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }
        }

        //Extra check on the number of extra nearest neighbors
        nENN *= (nENN > 0) ? 1 : 0;

        //The number of points in the neighborhood of the operator
        //\f$ = \text{monomial size} + \text{number of extra neighbors} \f$ nearest neighbors
        //Finding nearest neighbors
        L2NearestNeighbor<T> KNN(nPoints, nqPoints, nDim, msize + nENN);

        //Construct a kd-tree index & do nearest neighbors search
        KNN.buildIndex(idata, qdata);

        //Vector of all idata points' distance of their nearest neighbor
        T *idataminDist = nullptr;

        {
            //Finding only one nearest neighbor for the input data points
            L2NearestNeighbor<T> KNN1(nPoints, nDim, 1);

            //Construct a kd-tree index & do nearest neighbors search
            KNN1.buildIndex(idata);

            idataminDist = KNN1.minDist();
        }

        //Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
        //\f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
        EVectorX<T> B0 = EVectorX<T>::Zero(msize);
        B0(0) = static_cast<T>(1);

        //Total number of nearest neighbours for each point
        int nNN = KNN.numNearestNeighbors();

        //Array for keeping the component-wise L1 distances
        T L1Dist[nNN * nDim];

        //Creating a transpose of the Vandermonde matrix
        //with the size of monomials * monomials \f$  = l \times l \f$
        EMatrixX<T> VMT;
        VMT.resize(msize, nNN);

        //Matrix of exponential window function
        EVectorX<T> EM;
        EM.resize(nNN);

        //Matrix A of a linear system for the kernel coefficients
        EMatrixX<T> AM;
        AM.resize(msize, msize);

        //${\mathbf a}^T({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
        EVectorX<T> SV;
        SV.resize(msize);

        //Loop over all query points
        for (int i = 0; i < nqPoints; i++)
        {
            std::ptrdiff_t const IdI = i * nDim;

            //For each point \f$ {\mathbf x} \f$ we define \f$ \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, \f$
            //as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
            {
                //A pointer to nearest neighbors indices of point i
                int *nearestneighbors = KNN.nearestneighbors(i);

                //A pointer to the array for keeping the component-wise L1 distances
                T *l1dist = L1Dist;

                //\f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
                for (int j = 0; j < nNN; j++)
                {
                    std::ptrdiff_t const IdJ = nearestneighbors[j] * nDim;

                    //pointer to query data
                    T *Idata = qdata + IdI;

                    //pointer to idata (neighbors of i)
                    T *Jdata = idata + IdJ;

                    for (int d = 0; d < nDim; d++)
                    {
                        *l1dist++ = *Idata++ - *Jdata++;
                    }
                }
            }

            //Compute component-wise average neighbor spacing
            T h_avg(0);
            std::for_each(L1Dist, L1Dist + nNN * nDim, [&](T const l_i) { h_avg += std::abs(l_i); });

            //Component-wise average neighbor spacing \f$ h \f$
            h_avg /= static_cast<T>(nNN);

            //Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
            T const epsilon = h_avg / ratio;

            //Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
            std::for_each(L1Dist, L1Dist + nNN * nDim, [&](T &l_i) { l_i /= epsilon; });

            {
                //A pointer to nearest neighbors indices of point i
                int *nearestneighbors = KNN.nearestneighbors(i);

                //A pointer to nearest neighbors distances from the point i
                T *nnDist = KNN.NearestNeighborsDistances(i);

                //Primitive (quartic spline) object
                quartic_spline<T> q;

                //Loop through the neighbors
                for (int j = 0; j < nNN; j++)
                {
                    //Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    //Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    T column[msize];
                    poly.monomial_value(L1Dist + Id, column);

                    TEMapX<T, Eigen::ColMajor> columnV(column, msize, 1);

                    //Fill the Vandermonde matrix column by column
                    VMT.block(0, j, msize, 1) << columnV;

                    //Neighbor point number
                    std::ptrdiff_t const IdJ = nearestneighbors[j];

                    //Using a smooth correction function that satisfies
                    //\f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
                    //Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
                    //between the point and its nearest neighbors
                    T s = nnDist[j] / (0.9 * idataminDist[IdJ]);

                    //Index inside the kernel
                    std::ptrdiff_t const IdK = i * msize + j;

                    //Compute the kernel value at the point IdK
                    T kernelV = q.f(s);

                    //Assemble the right hand side
                    //\f$ {\mathbf b}={\mathbf P}({\mathbf x}) |_{{\mathbf x}=0} - \sum_{p} {\mathbf P}{\left(\frac{{\mathbf x}-{\mathbf x}_p}{\epsilon({\mathbf x})}\right)} {\mathbf C}\left(\frac{{\mathbf x}-{\mathbf x}_p}{c({\mathbf x}_p)} \right) \f$
                    B0 -= kernelV * columnV;

                    kernel[IdK] = kernelV;
                }

                T const epsilonsq = 2 * epsilon * epsilon;

                for (int j = 0; j < nNN; j++)
                {
                    EM(j) = std::exp(-nnDist[j] * nnDist[j] / epsilonsq);
                }
            }

            //\f$ \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
            //{\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
            //{\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
            //\end{matrix}
            EMatrixX<T> BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            //LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(BMT);

            //if necessary, remove redundant equations/coefficients
            if (lu.rank() < msize)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << "There are some singularities! we use a least-squares solution!" << std::endl;

                AM = BMT * BMT.transpose();

                Eigen::JacobiSVD<EMatrixX<T>> svd(AM);
                SV = svd.solve(B0);
            }
            else
            {
                EMatrixX<T> image = lu.image(BMT);
                AM = image * image.transpose();

                lu.compute(AM);

                SV = lu.solve(B0);
            }

            //SV contains the solution to the \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$

            {
                //A pointer to nearest neighbors indices of point i
                int *nearestneighbors = KNN.nearestneighbors(i);

                //A pointer to nearest neighbors distances from the point i
                T *nnDist = KNN.NearestNeighborsDistances(i);

                T const epsilonsq = epsilon * epsilon;

                //Loop through the neighbors
                for (int j = 0; j < nNN; j++)
                {
                    //Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    //Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    T column[msize];
                    poly.monomial_value(L1Dist + Id, column);

                    TEMapX<T, Eigen::ColMajor> columnV(column, msize, 1);

                    //Index inside the kernel
                    std::ptrdiff_t const IdK = i * msize + j;

                    T const expo = std::exp(-nnDist[j] * nnDist[j] / epsilonsq);
                    
                    kernel[IdK] += SV.dot(columnV) * expo;
                }
            }
        } //Loop over all points

        return true;
    }

  private:
    //Dimensiononality
    int nDim;
};

#endif
