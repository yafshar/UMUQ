TESTS=
check_PROGRAMS=

AM_CPPFLAGS = $(GTEST_CPPFLAGS)
AM_CXXFLAGS = $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

TESTS += polynomial_test
check_PROGRAMS += polynomial_test
polynomial_test_SOURCES = polynomial_test.cpp 

TESTS += parser_test
check_PROGRAMS += parser_test
parser_test_SOURCES = parser_test.cpp

TESTS += utility_test
check_PROGRAMS += utility_test
utility_test_SOURCES = utility_test.cpp

TESTS += TMCMC_datatype_test
check_PROGRAMS += TMCMC_datatype_test
TMCMC_datatype_test_SOURCES = TMCMC_datatype_test.cpp