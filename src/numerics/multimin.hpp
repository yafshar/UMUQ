#ifndef UMUQ_MULTIMIN_H
#define UMUQ_MULTIMIN_H

/*!
 * \file numerics/multimin.hpp
 * \brief Implementation of the Multidimensional Minimization.
 *
 * The multimin Module contains the c++ re-implamentation and modification 
 * to the original GSL Multidimensional Minimization source codes made  
 * available under the following license:
 * 
 * \verbatim
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 Fabrice Rossi
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 * \endverbatim
 */

/*! \namespace multimin
 * \brief Namespace containing all the functions for Multidimensional Minimization Module
 * 
 * It includes all the functionalities for finding minima of arbitrary multidimensional 
 * functions. It provides low level components for a variety of iterative minimizers 
 * and convergence tests.
 */
namespace multimin
{
#include "multimin/multimin_function.hpp"

#include "multimin/multimin_linear_minimize.hpp"
#include "multimin/multimin_directional_minimize.hpp"
#include "multimin/multimin_linear_wrapper.hpp"

#include "multimin/multimin_steepest_descent.hpp"
#include "multimin/multimin_conjugate_fr.hpp"
#include "multimin/multimin_conjugate_pr.hpp"
#include "multimin/multimin_vector_bfgs.hpp"
#include "multimin/multimin_vector_bfgs2.hpp"

#include "multimin/multimin_nmsimplex.hpp"
#include "multimin/multimin_nmsimplex2.hpp"
#include "multimin/multimin_nmsimplex2rand.hpp"
}

#endif
