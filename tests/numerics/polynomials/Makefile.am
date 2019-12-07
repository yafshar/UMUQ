TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

# numerics/polynomials
TESTS += polynomial_test
check_PROGRAMS += polynomial_test
polynomial_test_SOURCES = polynomial_test.cpp

TESTS += legendrepolynomial_test
check_PROGRAMS += legendrepolynomial_test
legendrepolynomial_test_SOURCES = legendrepolynomial_test.cpp

TESTS += hermitepolynomial_test
check_PROGRAMS += hermitepolynomial_test
hermitepolynomial_test_SOURCES = hermitepolynomial_test.cpp

TESTS += chebyshevpolynomial_test
check_PROGRAMS += chebyshevpolynomial_test
chebyshevpolynomial_test_SOURCES = chebyshevpolynomial_test.cpp
