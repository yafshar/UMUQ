#
# SYNOPSIS
#
#  AX_DEBUG([]])
#
# DESCRIPTION
#  Conditional that is true if the user uses the --enable-debug option
#
# AUTHOR
#  Yaser Afshar @ ya.afshar@gmail.com
#  Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_DEBUG], [AX_DEBUG])
AC_DEFUN([AX_DEBUG], [
  AC_ARG_ENABLE([debug],
    AS_HELP_STRING([--enable-debug@<:@=ARG@:>@],
        [Enable debug preprocessor flag (default is no) (optional)]),
    [
      if test x"${enableval}" = xno ; then
        ax_debug_ok=no
      elif test x"${enableval}" = xyes ; then
        ax_debug_ok=yes
      elif test x"${enableval}" != x ; then
        ax_debug_ok=no
      else
        ax_debug_ok=yes
      fi
    ], [
      ax_debug_ok=no
    ]
  )

  AM_CONDITIONAL([DEBUG], [test x"$ax_debug_ok" = xyes])
  AM_COND_IF([DEBUG], [
      CPPFLAGS+=" -DDEBUG"
    ], [
      AC_MSG_WARN([DEBUG preprocessor flag is not enabled !!!])
      CPPFLAGS+=" -DNDEBUG"
  ])
  AC_SUBST(CPPFLAGS)

  AC_MSG_RESULT()
])
