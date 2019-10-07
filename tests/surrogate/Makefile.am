TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

#surrogate
TESTS += polynomialtail_test
check_PROGRAMS += polynomialtail_test
polynomialtail_test_SOURCES = polynomialtail_test.cpp

TESTS += radialbasisfunctionkernel_test
check_PROGRAMS += radialbasisfunctionkernel_test
radialbasisfunctionkernel_test_SOURCES = radialbasisfunctionkernel_test.cpp

TESTS += radialbasisfunction_test
check_PROGRAMS += radialbasisfunction_test
radialbasisfunction_test_SOURCES = radialbasisfunction_test.cpp
