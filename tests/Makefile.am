TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

#numerics
TESTS += polynomial_test
check_PROGRAMS += polynomial_test
polynomial_test_SOURCES = numerics/polynomial_test.cpp 

TESTS += eigen_test
check_PROGRAMS += eigen_test
eigen_test_SOURCES = numerics/eigen_test.cpp 

TESTS += stats_test
check_PROGRAMS += stats_test
stats_test_SOURCES = numerics/stats_test.cpp 

TESTS += factorial_test
check_PROGRAMS += factorial_test
factorial_test_SOURCES = numerics/factorial_test.cpp 

TESTS += psrandom_test
check_PROGRAMS += psrandom_test
psrandom_test_SOURCES = numerics/random/psrandom_test.cpp 

TESTS += multimin_test
check_PROGRAMS += multimin_test 
multimin_test_SOURCES = numerics/multimin_test.cpp 

TESTS += knearestneighbors_test 
check_PROGRAMS += knearestneighbors_test 
knearestneighbors_test_SOURCES = numerics/knearestneighbors_test.cpp 

TESTS += dcpse_test
check_PROGRAMS += dcpse_test
dcpse_test_SOURCES = numerics/dcpse_test.cpp 

TESTS += fitness_test
check_PROGRAMS += fitness_test
fitness_test_SOURCES = numerics/fitness_test.cpp 

TESTS += linearregression_test
check_PROGRAMS += linearregression_test
linearregression_test_SOURCES = numerics/linearregression_test.cpp 

#numerics/function
TESTS += function_test
check_PROGRAMS += function_test
function_test_SOURCES = numerics/function/function_test.cpp 

TESTS += fitfunction_test
check_PROGRAMS += fitfunction_test
fitfunction_test_SOURCES = numerics/function/fitfunction_test.cpp 

TESTS += functionminimizer_test
check_PROGRAMS += functionminimizer_test
functionminimizer_test_SOURCES = numerics/function/functionminimizer_test.cpp

#numerics/density
TESTS += densityfunction_test
check_PROGRAMS += densityfunction_test
densityfunction_test_SOURCES = numerics/density/densityfunction_test.cpp 

#numerics/multimin
TESTS += conjugatefr_test
check_PROGRAMS += conjugatefr_test
conjugatefr_test_SOURCES = numerics/multimin/conjugatefr_test.cpp 

TESTS += conjugatepr_test
check_PROGRAMS += conjugatepr_test
conjugatepr_test_SOURCES = numerics/multimin/conjugatepr_test.cpp 

TESTS += simplexnm_test
check_PROGRAMS += simplexnm_test
simplexnm_test_SOURCES = numerics/multimin/simplexnm_test.cpp 

TESTS += simplexnm2_test
check_PROGRAMS += simplexnm2_test
simplexnm2_test_SOURCES = numerics/multimin/simplexnm2_test.cpp 

TESTS += simplexnm2rnd_test
check_PROGRAMS += simplexnm2rnd_test
simplexnm2rnd_test_SOURCES = numerics/multimin/simplexnm2rnd_test.cpp 

TESTS += steepestdescent_test
check_PROGRAMS += steepestdescent_test
steepestdescent_test_SOURCES = numerics/multimin/steepestdescent_test.cpp 

TESTS += bfgs_test
check_PROGRAMS += bfgs_test
bfgs_test_SOURCES = numerics/multimin/bfgs_test.cpp 

TESTS += bfgs2_test
check_PROGRAMS += bfgs2_test
bfgs2_test_SOURCES = numerics/multimin/bfgs2_test.cpp 

#misc
TESTS += utility_test
check_PROGRAMS += utility_test
utility_test_SOURCES = misc/utility_test.cpp

TESTS += parser_test
check_PROGRAMS += parser_test
parser_test_SOURCES = misc/parser_test.cpp

TESTS += arraywrapper_test
check_PROGRAMS += arraywrapper_test
arraywrapper_test_SOURCES = misc/arraywrapper_test.cpp

TESTS += funcallcounter_test
check_PROGRAMS += funcallcounter_test
funcallcounter_test_SOURCES = misc/funcallcounter_test.cpp

#io
TESTS += io_test
check_PROGRAMS += io_test
io_test_SOURCES = io/io_test.cpp

TESTS += eigen_io_test
check_PROGRAMS += eigen_io_test
eigen_io_test_SOURCES = io/eigen_io_test.cpp

TESTS += pyplot_test
check_PROGRAMS += pyplot_test
pyplot_test_SOURCES = io/pyplot_test.cpp

#data
TESTS += stdata_test
check_PROGRAMS += stdata_test
stdata_test_SOURCES = data/stdata_test.cpp

TESTS += runinfo_test
check_PROGRAMS += runinfo_test
runinfo_test_SOURCES = data/runinfo_test.cpp

TESTS += database_test
check_PROGRAMS += database_test
database_test_SOURCES = data/database_test.cpp

TESTS += datatype_test
check_PROGRAMS += datatype_test
datatype_test_SOURCES = data/datatype_test.cpp

#inference
#prior
TESTS += priordistribution_test
check_PROGRAMS += priordistribution_test
priordistribution_test_SOURCES = inference/prior/priordistribution_test.cpp

#tmcmc
TESTS += tmcmc_test
check_PROGRAMS += tmcmc_test
tmcmc_test_SOURCES = inference/tmcmc/tmcmc_test.cpp

#torc
TESTS += torc_test
check_PROGRAMS += torc_test
torc_test_SOURCES = torc/torc_test.cpp
