TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

# numerics/knn
TESTS += knearestneighbors_test
check_PROGRAMS += knearestneighbors_test
knearestneighbors_test_SOURCES = knearestneighbors_test.cpp