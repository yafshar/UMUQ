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
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_PYTHON], [AX_PYTHON])
AC_DEFUN([AX_PYTHON], [
	AC_ARG_ENABLE([python],
		AS_HELP_STRING([--enable-python@<:@=ARG@:>@], 
				[Enable python (default is no) (optional)]), 
		[
			if test x"${enableval}" = xno ; then
				ax_python_ok=no
			elif test x"${enableval}" = xyes ; then
				ax_python_ok=yes
			elif test x"${enableval}" != x ; then
				ax_python_ok=no
			else
				ax_python_ok=yes
			fi
		], [
			ax_python_ok=no
		]
	)

	AS_IF([test x"$ax_python_ok" = xyes], [
		AC_MSG_CHECKING(for python build information)
		AC_MSG_RESULT([])
		for python in python2.7 python python3.7 python3.6 python3.5 python3.4 python3.3 python3.2 python3.1 python3.0; do
			AC_CHECK_PROGS(PYTHON_BIN, [$python])
			ax_python_bin=$PYTHON_BIN
			if test x$ax_python_bin != x; then
				AC_CHECK_LIB($ax_python_bin, main, ax_python_lib=$ax_python_bin, ax_python_lib=no)
				if test x$ax_python_lib == xno; then
					AC_CHECK_LIB(${ax_python_bin}m, main, ax_python_lib=${ax_python_bin}m, ax_python_lib=no)
				fi
				if test x$ax_python_lib != xno; then
					ax_python_header=`$ax_python_bin -c "from distutils.sysconfig import *; print(get_config_var('CONFINCLUDEPY'))"`
					if test x$ax_python_header != x; then
						break;
					fi
				fi
			fi
		done
		if test x$ax_python_bin = x; then
			ax_python_bin=no
		fi
		if test x$ax_python_header = x; then
			ax_python_header=no
		fi
		if test x$ax_python_lib = x; then
			ax_python_lib=no
		fi

		AC_MSG_RESULT([Results of the Python check:])
		AC_MSG_RESULT([Binary:      $ax_python_bin])
		AC_MSG_RESULT([Library:     $ax_python_lib])
		AC_MSG_RESULT([Include Dir: $ax_python_header])

		if test x$ax_python_header != xno; then
			PYTHON_INCLUDE_DIR=$ax_python_header
			AC_SUBST(PYTHON_INCLUDE_DIR)
			CPPFLAGS+=" -I$ax_python_header"
			ax_python_ok=yes
		fi
		if test x$ax_python_lib != xno; then
			PYTHON_LIB=$ax_python_lib
			AC_SUBST(PYTHON_LIB)
		fi
	])

	AM_CONDITIONAL([HAVE_PYTHON], [test x"$ax_python_ok" = xyes])
	AM_COND_IF([HAVE_PYTHON], [
			AC_DEFINE(HAVE_PYTHON, 1, [Define if you want to use PYTHON.])
			AC_SUBST(CPPFLAGS)
		], [
			AC_MSG_WARN([You can not use pyplot class without PYTHON !!!])
	])

	AC_MSG_RESULT()
])


