AC_PREREQ([2.69])
AC_INIT([UMHBM],[1.0],[ya.afshar@gmail.com])

AC_CONFIG_AUX_DIR([auxdir])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_TESTDIR([tests])

AC_CANONICAL_BUILD

AC_CONFIG_HEADERS(src/umhbm_config.h)

AM_INIT_AUTOMAKE

AC_MSG_RESULT(---------------------------------------------)
AC_MSG_RESULT(---------  Configuring UMich HBM  -----------)
AC_MSG_RESULT(---------------------------------------------)

# AC_CONFIG_MACRO_DIR([m4])
m4_include([m4/m4_ax_cxx_compile_stdcxx.m4])
m4_include([m4/m4_ax_eigen.m4])
m4_include([m4/m4_ax_flann.m4])
m4_include([m4/m4_ax_blas.m4])
m4_include([m4/m4_ax_lapack.m4])
m4_include([m4/m4_ax_googletest.m4])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for libraries.

### Checking for EIGEN
AC_LANG_PUSH([C++])
AC_MSG_CHECKING([ Checking for EIGEN Library ])
AC_CHECK_HEADERS([Eigen/Dense], [ax_eigen_ok=yes], [ax_eigen_ok=no])
AS_IF([test x$ax_eigen_ok != xyes], [ AX_EIGEN ])
AS_IF([test x$ax_eigen_ok != xyes], [ AC_MSG_ERROR([ Unable to find the EIGEN library !])])
AC_LANG_POP

### Checking for FLANN
AC_LANG_PUSH([C++])
AC_MSG_CHECKING([ Checking for FLANN Library ])
AC_CHECK_HEADERS([flann/flann.h], [ax_flann_ok=yes], [ax_flann_ok=no])
AS_IF([test x$ax_flann_ok != xyes], [ AX_FLANN ])
AS_IF([test x$ax_flann_ok != xyes], [ AC_MSG_ERROR([ Unable to find the FLANN library !])])
AC_LANG_POP

### Checking for LAPACK
AC_MSG_CHECKING([ Checking for LAPACK Library ])
AX_LAPACK
AS_IF([test x$ax_lapack_ok != xyes], [ AC_MSG_ERROR([ Unable to find the LAPACK library !])])

### Checking for GOOGLETEST
AC_MSG_CHECKING([ Checking for GOOGLETEST Framework ])
AX_GOOGLETEST

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE

# use the C++ compiler for the following checks
AC_LANG([C++])

# ensure we are using c11 C++ standard
AX_CXX_COMPILE_STDCXX( [11], [noext], [mandatory])

# Checks for library functions.
AC_CHECK_FUNCS([pow])

AC_CONFIG_FILES([Makefile src/Makefile tests/Makefile])
AC_OUTPUT

AC_MSG_RESULT([
DCPSE Surrogate $VERSION is now configured
Configure Information:
  C Compiler        : $CC
    DEFS            :   $DEFS
    CPPFLAGS        :   $CPPFLAGS
    CFLAGS          :   $CFLAGS

  C++ Compiler      : $CXX
    DEFS            :   $DEFS
    CPPFLAGS        :   $CPPFLAGS
    CXXFLAGS        :   $CXXFLAGS

  Linker            : $LD
    LDFLAGS         :   $LDFLAGS
    LIBS            :   $LIBS

  Google Test 
    GTEST_CPPFLAGS  : $GTEST_CPPFLAGS
    GTEST_CXXFLAGS  : $GTEST_CXXFLAGS
    GTEST_LDFLAGS   : $GTEST_LDFLAGS
    GTEST_LIBS      : $GTEST_LIBS
])
