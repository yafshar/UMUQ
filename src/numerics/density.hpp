#ifndef UMUQ_DENSITY_H
#define UMUQ_DENSITY_H

/*!
 * \file numerics/density.hpp
 * \brief Implementation of the collections of density functions (probability density (PDF))
 * 
 * This module includes:
 * - \b densityFunction                  - A base density function or a probability density (PDF) function class  \sa densityFunction
 * 
 * - \b uniformDistribution              - Flat (Uniform) distribution function
 * - \b gaussianDistribution             - The Gaussian distribution
 * - \b multivariategaussianDistribution - The Multivariate Gaussian Distribution (generalization of the one-dimensional (univariate) Gaussian ) \sa multivariategaussianDistribution
 * - \b multivariateGaussianDistribution - The Multivariate Gaussian Distribution (generalization of the one-dimensional (univariate) Gaussian ) \sa multivariateGaussianDistribution
 * - \b exponentialDistribution          - The exponential distribution \sa exponentialDistribution
 * - \b gammaDistribution                - The Gamma distribution \sa gammaDistribution
 * - \b multinomialDistribution          - The multinomial distribution \sa multinomialDistribution
 */


#include "function/densityfunction.hpp"

#include "density/uniformdistribution.hpp"
#include "density/exponentialdistribution.hpp"
#include "density/gammadistribution.hpp"
#include "density/gaussiandistribution.hpp"
#include "density/multivariategaussiandistribution.hpp"
#include "density/multinomialdistribution.hpp"

#endif // UMUQ_DENSITY
