AC_PREREQ([2.69])
AC_INIT([DCPSE Surrogate], [1.0], [ya.afshar@gmail.com])
AC_CONFIG_SRCDIR([src])
AM_INIT_AUTOMAKE

AC_MSG_RESULT(---------------------------------------------)
AC_MSG_RESULT(------- Configuring DCPSE Surrogate ---------)
AC_MSG_RESULT(---------------------------------------------)

# AC_CONFIG_MACRO_DIR([m4])
m4_include([m4/m4_ax_cxx_compile_stdcxx.m4])
m4_include([m4/m4_ax_eigen.m4])
m4_include([m4/m4_ax_flann.m4])

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

AC_OUTPUT(Makefile src/Makefile)
