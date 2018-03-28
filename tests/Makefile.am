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


#io
TESTS += io_test
check_PROGRAMS += io_test
io_test_SOURCES = io/io_test.cpp


#data_type
TESTS += TMCMC_datatype_test
check_PROGRAMS += TMCMC_datatype_test
TMCMC_datatype_test_SOURCES = data_type/TMCMC_datatype_test.cpp
