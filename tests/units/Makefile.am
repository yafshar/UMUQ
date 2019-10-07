TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

#units
TESTS += species_test
check_PROGRAMS += species_test
species_test_SOURCES = species_test.cpp

TESTS += units_test
check_PROGRAMS += units_test
units_test_SOURCES = units_test.cpp

TESTS += lattice_test
check_PROGRAMS += lattice_test
lattice_test_SOURCES = lattice_test.cpp
