#ifndef UMUQ_MULTIMIN_H
#define UMUQ_MULTIMIN_H

#include "core/core.hpp"

/*!
 * \file numerics/multimin.hpp
 * \ingroup Multimin_Module
 *
 * \brief Implementation of the Multidimensional Minimization.
 *
 * The multimin Module contains the c++ re-implamentation and modification
 * to the original GSL Multidimensional Minimization source codes made
 * available under the following license:
 *
 * \copyright
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

#include "datatype/functionminimizertype.hpp"
#include "datatype/differentiablefunctionminimizertype.hpp"
#include "datatype/functiontype.hpp"

#include "function/functionminimizer.hpp"
#include "function/differentiablefunctionminimizer.hpp"

#include "multimin/steepestdescent.hpp"
#include "multimin/conjugatefr.hpp"
#include "multimin/conjugatepr.hpp"
#include "multimin/bfgs.hpp"
#include "multimin/bfgs2.hpp"

#include "multimin/simplexnm.hpp"
#include "multimin/simplexnm2.hpp"
#include "multimin/simplexnm2rnd.hpp"

#endif // UMUQ_MULTIMIN
