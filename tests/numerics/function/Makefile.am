TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

#numerics/function
TESTS += function_test
check_PROGRAMS += function_test
function_test_SOURCES = function_test.cpp

TESTS += fitfunction_test
check_PROGRAMS += fitfunction_test
fitfunction_test_SOURCES = fitfunction_test.cpp

TESTS += functionminimizer_test
check_PROGRAMS += functionminimizer_test
functionminimizer_test_SOURCES = functionminimizer_test.cpp
