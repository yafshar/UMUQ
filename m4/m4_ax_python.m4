# ===========================================================================
#        https://www.gnu.org/software/autoconf-archive/ax_python.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_PYTHON([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro does a complete Python development environment check.
#
#   It checks for all known versions. When it finds an executable, it looks
#   to find the header files and library.
#
#   It sets PYTHON_BIN to the name of the python executable,
#   PYTHON_INCLUDE_DIR to the directory holding the header files, and
#   PYTHON_LIB to the name of the Python library.
#
#   This macro calls AC_SUBST on PYTHON_BIN (via AC_CHECK_PROG),
#   PYTHON_INCLUDE_DIR and PYTHON_LIB.
#
# LICENSE
#
#   Copyright (c) 2008 Michael Tindal
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <https://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.
#
# ADAPTED
#  Yaser Afshar @ ya.afshar@gmail.com
#  Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_PYTHON], [AX_PYTHON])
AC_DEFUN([AX_PYTHON], [
  AC_ARG_ENABLE([python],
    AS_HELP_STRING([--enable-python@<:@=ARG@:>@],
        [Enable python (default is no) (optional). it is possible to specify the version, or the root directory for PYTHON]),
    [
      PYTHON=
      PYTHON_VERSION=
      if test x"${enableval}" = xno ; then
        ax_python_ok=no
      elif test x"${enableval}" = xyes ; then
        ax_python_ok=yes
      elif test x"${enableval}" != x ; then
        for python_version in 2.7; do
          if test x"${enableval}" = x"${python_version}"; then
            PYTHON_VERSION=${python_version}
            AC_PATH_PROG([PYTHON], [python[${PYTHON_VERSION}]])
            if test -z "${PYTHON}"; then
              PYTHON_VERSION=""
              AC_MSG_ERROR([Cannot find the requested python version ${PYTHON_VERSION} in system path!])
            fi
            break;
          elif test x"${enableval}" = x"python${python_version}" ; then
            PYTHON_VERSION=${python_version}
            AC_PATH_PROG([PYTHON], [python[${PYTHON_VERSION}]])
            if test -z "${PYTHON}"; then
              PYTHON_VERSION=""
              AC_MSG_ERROR([Cannot find the requested python version ${PYTHON_VERSION} in system path!])
            fi
            break;
          fi
        done
        if test x"${enableval}" = x"python"; then
          AC_PATH_PROG([PYTHON], [python])
          if test -z "${PYTHON}"; then
            PYTHON_VERSION=""
            AC_MSG_ERROR([Cannot find python in the system path!])
          fi
          PYTHON_VERSION=`${PYTHON} -c "import sys; print(sys.version.split()[[0][:3]])"`
        fi
        if test -x "${enableval}"; then
          # $enableval must be the executable path
          AC_SUBST([PYTHON], ["${enableval}"])
          PYTHON_VERSION=`${PYTHON} -c "import sys; print(sys.version.split()[[0][:3]])"`
        fi
        if test -n "${PYTHON}"; then
          if test -n "${PYTHON_VERSION}"; then
            ax_python_ok=yes
          else
            ax_python_ok=no
          fi
        else
          ax_python_ok=no
        fi
      else
        ax_python_ok=yes
      fi
    ], [
      ax_python_ok=no
    ]
  )

  AS_IF([test x"${ax_python_ok}" = xyes], [
    AC_MSG_CHECKING(for python build information)
    AC_MSG_RESULT([])

    succeeded=no

    PYTHON_LD_LIBRARY_PATH=
    PYTHON_LDFLAGS=
    PYTHON_CPPFLAGS=
    PYTHON_PATH=
    PYTHON_EXECPATH=

    if test -z "${PYTHON}"; then
      for python_version in 2.7; do
        AC_PATH_PROG([PYTHON], [python[$python_version]])
        if test -n "${PYTHON}"; then
          PYTHON_VERSION=`${PYTHON} -c "import sys; print(sys.version.split()[[0][:3]])"`
          break;
        fi
      done
    fi

    if test -z "${PYTHON}"; then
      AC_MSG_ERROR([Cannot find python in the system path!])
    fi

    PYTHON_MAJOR_VERSION=`${PYTHON} -c "import sys; print(sys.version.split()[[0][:1]])"`

    PYTHON_LDVERSION=`${PYTHON} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LDVERSION'))"`
    if test x"${PYTHON_LDVERSION}" = x"None"; then
      PYTHON_LDVERSION="${PYTHON_VERSION}"
    fi

    PYTHON_PATH=`${PYTHON} -c "import sys; print(sys.prefix)"`
    if test -z "$PYTHON_PATH"; then
      AC_MSG_ERROR([Python prefix is not known!])
    fi

    PYTHON_INCLUDE=`${PYTHON} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('CONFINCLUDEPY'))"`
    if test -z "${PYTHON_INCLUDE}"; then
      if test -d "${PYTHON_PATH}/include/python${PYTHON_LDVERSION}" && test -r "${PYTHON_PATH}/include/python${PYTHON_LDVERSION}" ; then
        PYTHON_CPPFLAGS="-I${PYTHON_PATH}/include/python${PYTHON_LDVERSION}"
      fi
    else
      PYTHON_CPPFLAGS='-I'"${PYTHON_INCLUDE}"
    fi

    PYTHON_EXECPATH=`${PYTHON} -c "import sys; print(sys.exec_prefix)"`
    if test x"${PYTHON_PATH}" != x"${PYTHON_EXECPATH}"; then
      if test -d "${PYTHON_EXECPATH}/include/python${PYTHON_LDVERSION}" && test -r "${PYTHON_EXECPATH}/include/python${PYTHON_LDVERSION}" ; then
        PYTHON_INCLUDE+=" ${PYTHON_EXECPATH}/include/python${PYTHON_LDVERSION}"
        PYTHON_CPPFLAGS+=" -I${PYTHON_EXECPATH}/include/python${PYTHON_LDVERSION}"
      fi
    fi

    PYTHON_LD_LIBRARY_PATH=`${PYTHON} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LIBDIR'))"`
    PYTHON_LDFLAGS=" -L${PYTHON_LD_LIBRARY_PATH}"
    PYTHON_LIB='python'"${PYTHON_LDVERSION}"

    CPPFLAGS_SAVED="${CPPFLAGS}"
    LDFLAGS_SAVED="${LDFLAGS}"
    LIBS_SAVED="${LIBS}"

    CPPFLAGS+=" ${PYTHON_CPPFLAGS}"
    LDFLAGS+="${PYTHON_LDFLAGS}"' -l'"${PYTHON_LIB}"

    AC_LANG_PUSH([C++])
    AC_CHECK_LIB(${PYTHON_LIB}, PyArg_ParseTuple,
      [
        LDFLAGS+=' -Wl,-rpath,'"$PYTHON_LD_LIBRARY_PATH"
        succeeded=yes
      ], [
        succeeded=no
      ]
    )
    AS_IF([test x"${LIBS}" = x"$LIBS_SAVED"], [LIBS='-l'"${PYTHON_LIB}"" ${LIBS}"])

    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
        @%:@include <Python.h>
      ]], [[]]
      )], [
        AC_MSG_RESULT(checking Python.h usability...  yes)
        AC_MSG_RESULT(checking Python.h presence... yes)
        AC_MSG_RESULT(checking for Python.h... yes)
        succeeded=yes
      ], [
        AC_MSG_RESULT(checking Python.h usability...  no)
        AC_MSG_RESULT(checking Python.h presence... no)
        AC_MSG_RESULT(checking for Python.h... no)
        succeeded=no
      ]
    )
    AC_LANG_POP([C++])

    if test x"$succeeded" = xyes ; then
      AC_SUBST([PYTHON_BIN], [${PYTHON}])
      AC_SUBST([PYTHON_LIB], [${PYTHON_LIB}])
      AC_SUBST([PYTHON_INCLUDE_DIR], [${PYTHON_INCLUDE}])
      AC_SUBST(PYTHON_LD_LIBRARY_PATH)
      ax_python_ok="yes"
      :
    else
      ax_python_ok="no"
      CPPFLAGS="$CPPFLAGS_SAVED"
      LDFLAGS="$LDFLAGS_SAVED"
      LIBS="$LIBS_SAVED"
      :
    fi
  ])

  ax_numpy_ok=no
  if test x"${ax_python_ok}" = xyes; then
    AC_MSG_CHECKING(for numpy python module)
    ${PYTHON} -c "import numpy" 2>/dev/null
    if test $? == 0; then
      AC_MSG_RESULT(found)
      ax_numpy_ok=yes
      AC_CACHE_CHECK([for numpy include directory], [_cv_numpy_header], [_cv_numpy_header=`${PYTHON} -c "import numpy; numpypath=numpy.__path__[[0]]; print('%s/core/include' % numpypath)"`])
      AC_SUBST([NUMPY_INCLUDE_DIR], [$_cv_numpy_header])
    else
      AC_MSG_WARN([ Unable to find NUMPY !])
    fi
  fi

  AM_CONDITIONAL([HAVE_PYTHON], [test x"${ax_numpy_ok}" = xyes])
  AM_COND_IF([HAVE_PYTHON], [
      AC_DEFINE(HAVE_PYTHON, 1, [Define if you want to use PYTHON.])
      AC_DEFINE_UNQUOTED(PYTHON_BIN, [${PYTHON}], [PYTHON executable.])
      AC_DEFINE_UNQUOTED(PYTHON_MAJOR_VERSION, [${PYTHON_MAJOR_VERSION}], [PYTHON Major version.])

      if test x${NUMPY_INCLUDE_DIR} != x; then
        CPPFLAGS+=" -I${NUMPY_INCLUDE_DIR}"
      fi

      AC_SUBST(CPPFLAGS)
      AC_SUBST(LDFLAGS)
    ], [
      AC_MSG_WARN([You can not use pyplot class without PYTHON !!!])
  ])

  AC_MSG_RESULT()
])
