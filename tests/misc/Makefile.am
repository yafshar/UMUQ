TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

#misc
TESTS += utility_test
check_PROGRAMS += utility_test
utility_test_SOURCES = utility_test.cpp

TESTS += parser_test
check_PROGRAMS += parser_test
parser_test_SOURCES = parser_test.cpp

TESTS += arraywrapper_test
check_PROGRAMS += arraywrapper_test
arraywrapper_test_SOURCES = arraywrapper_test.cpp

TESTS += funcallcounter_test
check_PROGRAMS += funcallcounter_test
funcallcounter_test_SOURCES = funcallcounter_test.cpp

TESTS += timer_test
check_PROGRAMS += timer_test
timer_test_SOURCES = timer_test.cpp
