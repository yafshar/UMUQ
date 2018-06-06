export LD_LIBRARY_PATH=${FLANN_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${FLANN_LD_LIBRARY_PATH}:$DYLD_LIBRARY_PATH

TESTS=
check_PROGRAMS=

AM_CPPFLAGS = $(GTEST_CPPFLAGS)
AM_CXXFLAGS = $(GTEST_CXXFLAGS)
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
psrandom_test_SOURCES = numerics/psrandom_test.cpp 

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

#misc
TESTS += utility_test
check_PROGRAMS += utility_test
utility_test_SOURCES = misc/utility_test.cpp

TESTS += parser_test
check_PROGRAMS += parser_test
parser_test_SOURCES = misc/parser_test.cpp

TESTS += array_test
check_PROGRAMS += array_test
array_test_SOURCES = misc/array_test.cpp

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


#data
TESTS += stdata_test
check_PROGRAMS += stdata_test
stdata_test_SOURCES = data/stdata_test.cpp

TESTS += runinfo_test
check_PROGRAMS += runinfo_test
runinfo_test_SOURCES = data/runinfo_test.cpp
