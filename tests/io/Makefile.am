TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

clean-local:
	rm -fr *.png *.svg

#io
TESTS += io_test
check_PROGRAMS += io_test
io_test_SOURCES = io_test.cpp

TESTS += eigen_io_test
check_PROGRAMS += eigen_io_test
eigen_io_test_SOURCES = eigen_io_test.cpp

TESTS += pyplot_test
check_PROGRAMS += pyplot_test
pyplot_test_SOURCES = pyplot_test.cpp
