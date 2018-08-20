#
# SYNOPSIS
#
#	AX_DEBUG([]])
#
# DESCRIPTION
#	Conditional that is true if the user uses the --enable-debug option
#
# AUTHOR
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan

AC_DEFUN([AX_DEBUG], [
AC_ARG_ENABLE([debug],
	[  --enable-debug    Turn on debugging],
	[case "${enableval}" in
		yes) debug=true ;;
		no)  debug=false ;;
		*) AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
		esac],[debug=false])

	AM_CONDITIONAL([DEBUG], [test x$debug = xtrue])
	AM_COND_IF([DEBUG], [
			AC_DEFINE(DEBUG, 1, [Define if you enable debugging.])
		]
	)
])
