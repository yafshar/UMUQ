SUBDIRS = polynomials random hypercube function density multimin

TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

clean-local:
	rm -fr *.svg Ndata Xdata Qdata

#numerics

# #numerics/polynomials
# TESTS += polynomial_test
# check_PROGRAMS += polynomial_test
# polynomial_test_SOURCES = polynomials/polynomial_test.cpp

# TESTS += legendrepolynomial_test
# check_PROGRAMS += legendrepolynomial_test
# legendrepolynomial_test_SOURCES = polynomials/legendrepolynomial_test.cpp

# TESTS += hermitepolynomial_test
# check_PROGRAMS += hermitepolynomial_test
# hermitepolynomial_test_SOURCES = polynomials/hermitepolynomial_test.cpp

# TESTS += chebyshevpolynomial_test
# check_PROGRAMS += chebyshevpolynomial_test
# chebyshevpolynomial_test_SOURCES = polynomials/chebyshevpolynomial_test.cpp

TESTS += eigen_test
check_PROGRAMS += eigen_test
eigen_test_SOURCES = eigen_test.cpp

TESTS += stats_test
check_PROGRAMS += stats_test
stats_test_SOURCES = stats_test.cpp

TESTS += factorial_test
check_PROGRAMS += factorial_test
factorial_test_SOURCES = factorial_test.cpp

# #numerics/random
# TESTS += psrandom_test
# check_PROGRAMS += psrandom_test
# psrandom_test_SOURCES = random/psrandom_test.cpp

TESTS += multimin_test
check_PROGRAMS += multimin_test
multimin_test_SOURCES = multimin_test.cpp

TESTS += knearestneighbors_test
check_PROGRAMS += knearestneighbors_test
knearestneighbors_test_SOURCES = knearestneighbors_test.cpp

# #numerics/hypercube
# TESTS += hypercubesampling_test
# check_PROGRAMS += hypercubesampling_test
# hypercubesampling_test_SOURCES = hypercube/hypercubesampling_test.cpp

TESTS += dcpse_test
check_PROGRAMS += dcpse_test
dcpse_test_SOURCES = dcpse_test.cpp

TESTS += fitness_test
check_PROGRAMS += fitness_test
fitness_test_SOURCES = fitness_test.cpp

TESTS += linearregression_test
check_PROGRAMS += linearregression_test
linearregression_test_SOURCES = linearregression_test.cpp

# #numerics/function
# TESTS += function_test
# check_PROGRAMS += function_test
# function_test_SOURCES = function/function_test.cpp

# TESTS += fitfunction_test
# check_PROGRAMS += fitfunction_test
# fitfunction_test_SOURCES = function/fitfunction_test.cpp

# TESTS += functionminimizer_test
# check_PROGRAMS += functionminimizer_test
# functionminimizer_test_SOURCES = function/functionminimizer_test.cpp

# #numerics/density
# TESTS += densityfunction_test
# check_PROGRAMS += densityfunction_test
# densityfunction_test_SOURCES = density/densityfunction_test.cpp

# #numerics/multimin
# TESTS += conjugatefr_test
# check_PROGRAMS += conjugatefr_test
# conjugatefr_test_SOURCES = multimin/conjugatefr_test.cpp

# TESTS += conjugatepr_test
# check_PROGRAMS += conjugatepr_test
# conjugatepr_test_SOURCES = multimin/conjugatepr_test.cpp

# TESTS += simplexnm_test
# check_PROGRAMS += simplexnm_test
# simplexnm_test_SOURCES = multimin/simplexnm_test.cpp

# TESTS += simplexnm2_test
# check_PROGRAMS += simplexnm2_test
# simplexnm2_test_SOURCES = multimin/simplexnm2_test.cpp

# TESTS += simplexnm2rnd_test
# check_PROGRAMS += simplexnm2rnd_test
# simplexnm2rnd_test_SOURCES = multimin/simplexnm2rnd_test.cpp

# TESTS += steepestdescent_test
# check_PROGRAMS += steepestdescent_test
# steepestdescent_test_SOURCES = multimin/steepestdescent_test.cpp

# TESTS += bfgs_test
# check_PROGRAMS += bfgs_test
# bfgs_test_SOURCES = multimin/bfgs_test.cpp

# TESTS += bfgs2_test
# check_PROGRAMS += bfgs2_test
# bfgs2_test_SOURCES = multimin/bfgs2_test.cpp
