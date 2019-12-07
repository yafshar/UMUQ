#ifndef UMUQ_DENSITY_H
#define UMUQ_DENSITY_H

#include "core/core.hpp"

/*!
 * \file numerics/density.hpp
 * \brief Implementation of the collections of density functions (probability density (PDF))
 *
 * This module includes:
 * - \b densityFunction                  - A base density function or a probability density (PDF) function class  \sa umuq::density::densityFunction
 *
 * - \b uniformDistribution              - Flat (Uniform) distribution function \sa umuq::density::uniformDistribution
 * - \b gaussianDistribution             - The Gaussian distribution \sa umuq::density::gaussianDistribution
 * - \b multivariategaussianDistribution - The Multivariate Gaussian Distribution (generalization of the one-dimensional (univariate) Gaussian ) \sa umuq::density::multivariategaussianDistribution
 * - \b multivariateGaussianDistribution - The Multivariate Gaussian Distribution (generalization of the one-dimensional (univariate) Gaussian ) \sa umuq::density::multivariateGaussianDistribution
 * - \b exponentialDistribution          - The exponential distribution \sa umuq::density::exponentialDistribution
 * - \b gammaDistribution                - The Gamma distribution \sa umuq::density::gammaDistribution
 * - \b multinomialDistribution          - The multinomial distribution \sa umuq::density::multinomialDistribution
 */


#include "numerics/function/densityfunction.hpp"

#include "density/uniformdistribution.hpp"
#include "density/exponentialdistribution.hpp"
#include "density/gammadistribution.hpp"
#include "density/gaussiandistribution.hpp"
#include "density/multivariategaussiandistribution.hpp"
#include "density/multinomialdistribution.hpp"

#endif // UMUQ_DENSITY
