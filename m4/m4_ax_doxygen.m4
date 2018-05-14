#
# SYNOPSIS
#
#   AX_DOXYGEN([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   Test to see if the DOXYGEN has been installed.
#
# AUTHOR 
#  Yaser Afshar @ ya.afshar@gmail.com
#  Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_DOXYGEN], [AX_DOXYGEN])
AC_DEFUN([AX_DOXYGEN], [
	AC_ARG_ENABLE([doxygen], 
		AS_HELP_STRING([--enable-doxygen@<:@=ARG@:>@],
				[Enable doxygen (default is no) (optional)]), 
		[
			if test x"$enableval" = xno ; then
				AC_MSG_WARN([You can not generate software reference documentation without DOXYGEN !!!])
				ax_doxygen_ok=no
			elif test x"$enableval" = xyes ; then
				ax_doxygen_ok=yes
			elif test x"$enableval" != x ; then
				ax_doxygen_ok=yes
			else 
				AC_MSG_WARN([You can not generate software reference documentation without DOXYGEN !!!])
				ax_doxygen_ok=no
			fi
		],[
			ax_doxygen_ok=no
		]
	)

	AS_IF([test x"$ax_doxygen_ok" = xyes], [
		AC_MSG_NOTICE(DOXYGEN)

		AC_CHECK_PROGS([DOXYGEN], [doxygen])
		AC_CHECK_PROGS([DOT], [dot])
		AC_CHECK_PROGS([PDFLATEX], [pdflatex])

		if test -z "$DOT"; then
			AC_MSG_WARN([Doxygen needs dot, please install dot first !!!])
			ax_doxygen_ok=no
		fi
		if test -z "$PDFLATEX"; then
			AC_MSG_WARN([Doxygen needs pdflatex program !!!, it is part of TeX http://www.tug.org/texlive/acquire-netinstall.html])
			ax_doxygen_ok=no
		fi

		if test -z "$DOXYGEN"; then 
			AC_MSG_WARN([Doxygen not found !!! - continuing without Doxygen support.])
			ax_doxygen_ok=no
		fi

		AS_IF([test x"$ax_doxygen_ok" = xyes], [
			AC_DEFINE(HAVE_DOXYGEN, 1, [Define if you have DOXYGEN.])
		])
	])

	AM_CONDITIONAL([HAVE_DOXYGEN], [test x"$ax_doxygen_ok" = xyes])
	AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Doxyfile])], [AC_MSG_WARN([You can not generate software reference documentation without DOXYGEN !!!])])

	AC_MSG_RESULT()
])
