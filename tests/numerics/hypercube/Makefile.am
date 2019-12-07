TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

#numerics/hypercube
TESTS += hypercubesampling_test
check_PROGRAMS += hypercubesampling_test
hypercubesampling_test_SOURCES = hypercubesampling_test.cpp
