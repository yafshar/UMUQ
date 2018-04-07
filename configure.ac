#
# AUTHOR
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan
#

AC_INIT([UMHBM], [1.0], [ya.afshar@gmail.com])

# AC_COPYRIGHT(Copyright 2018 University of Michigan)
# AC_REVISION([$Revision: 1.00 $])

AC_PREREQ([2.69])

AC_CONFIG_AUX_DIR([auxdir])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_TESTDIR([tests])

AC_CONFIG_MACRO_DIR([m4])
# 
m4_include([m4/m4_ax_cxx_compile_stdcxx.m4])
m4_include([m4/m4_ax_mpi.m4])
m4_include([m4/m4_ax_pthread.m4])
m4_include([m4/m4_ax_torc.m4])
m4_include([m4/m4_ax_eigen.m4])
m4_include([m4/m4_ax_flann.m4])
m4_include([m4/m4_ax_blas.m4])
m4_include([m4/m4_ax_lapack.m4])
m4_include([m4/m4_ax_googletest.m4])
m4_include([m4/m4_ax_doxygen.m4])

# Configuring
AC_MSG_RESULT(---------------------------------------------)
AC_MSG_RESULT(---------  Configuring UMich HBM  -----------)
AC_MSG_RESULT(---------------------------------------------)

AC_CANONICAL_BUILD
AC_CANONICAL_HOST
AC_CANONICAL_TARGET
AS_IF([test "$host" != "$target"],
	[
		AC_MSG_WARN([Cross-compile detected])
		AC_MSG_WARN([Proceed at your own risk!])
	]
)

# Check for architectures with no support
case "${host}" in
	powerpc-*|powerpc64-*|ppc-*)
	AC_MSG_ERROR([Big endian PPC is not supported.])
	;;
esac
case "${target}" in
	powerpc-*|powerpc64-*|ppc-*)
	AC_MSG_ERROR([Big endian PPC is no longer supported.])
	;;
esac

AC_CONFIG_HEADERS(src/umhbm_config.h)

#
# Init automake
#
AM_INIT_AUTOMAKE([1.9 dist-bzip2 dist-zip foreign subdir-objects])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_F77_LIBRARY_LDFLAGS

AC_PROG_LIBTOOL

# Checks for libraries.
AC_MSG_RESULT() 

### Checking for MPI
AC_MSG_CHECKING([for - MPI Library - ])
AC_MSG_NOTICE(MPI)
AX_MPI

### Checking for EIGEN
AC_MSG_CHECKING([for - EIGEN Library - ])
AC_MSG_NOTICE(EIGEN)
AX_EIGEN

### Checking for FLANN
AC_MSG_CHECKING([for - FLANN Library - ])
AC_MSG_NOTICE(FLANN)
AX_FLANN

### Checking for BLAS
AC_MSG_CHECKING([for - BLAS Library - ])
AX_BLAS

### Checking for LAPACK
AC_MSG_CHECKING([for - LAPACK Library - ])
AX_LAPACK

### Checking for POSIX threads
AC_MSG_CHECKING([for - POSIX threads - ])
AX_PTHREAD

### Checking for GOOGLETEST
AC_MSG_CHECKING([for - GOOGLETEST Framework - ])
AX_GOOGLETEST

### Checking for TORC
AC_MSG_CHECKING([for - TORC Library - ])
AX_TORC

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE

AC_LANG_PUSH([C++])

# Ensure we are using c11 C++ standard.
AX_CXX_COMPILE_STDCXX([11], [noext], [mandatory])

# Checks for library functions.
AC_CHECK_FUNCS([pow])

AC_LANG_POP([C++])

# Chech for Doxygen
AC_MSG_CHECKING([ for - DOXYGEN - ])
AX_DOXYGEN

AC_CONFIG_FILES([Makefile src/Makefile])
AM_COND_IF([HAVE_GOOGLETEST], [AC_CONFIG_FILES([tests/Makefile])])
AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Makefile])])

AC_OUTPUT

AC_MSG_RESULT([
UMich HBM $VERSION is now configured

Configure Information:
  C preprocessor    : $CPP
    CPPFLAGS        :   $CPPFLAGS

  C Compiler        : $CC
    DEFS            :   $DEFS
    CFLAGS          :   $CFLAGS

  C++ Compiler      : $CXX
    DEFS            :   $DEFS
    CXXFLAGS        :   $CXXFLAGS

  Linker            : $LD
    LDFLAGS         :   $LDFLAGS
    LIBS            :   $LIBS

  MPI C   compiler  : $MPICC
  MPI C++ compiler  : $MPICXX
    MPILIBS         :   $MPILIBS

  Google Test 
    GTEST_CPPFLAGS  : $GTEST_CPPFLAGS
    GTEST_CXXFLAGS  : $GTEST_CXXFLAGS
    GTEST_LDFLAGS   : $GTEST_LDFLAGS
    GTEST_LIBS      : $GTEST_LIBS
])
