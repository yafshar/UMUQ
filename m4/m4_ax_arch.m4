#
# SYNOPSIS
#
#  AX_ARCH([]])
#
# DESCRIPTION
#  Check for architectures
#
# AUTHOR
#  Yaser Afshar @ ya.afshar@gmail.com
#  Dept of Aerospace Engineering | University of Michigan

AC_DEFUN([AX_ARCH], [
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
])
