# ===========================================================================
#        https://www.gnu.org/software/autoconf-archive/ax_lapack.html
# ===========================================================================
#
# SYNOPSIS
#
#  AX_LAPACK([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#  This macro looks for a library that implements the LAPACK linear-algebra
#  interface (see http://www.netlib.org/lapack/). On success, it sets the
#  LAPACK_LIBS output variable to hold the requisite library linkages.
#
#  To link with LAPACK, you should link with:
#
#  $LAPACK_LIBS $BLAS_LIBS $LIBS $FCLIBS
#
#  in that order. BLAS_LIBS is the output variable of the AX_BLAS macro,
#  called automatically. FCLIBS is the output variable of the
#  AC_FC_LIBRARY_LDFLAGS macro (called if necessary by AX_BLAS), and is
#  sometimes necessary in order to link with FC libraries. Users will also
#  need to use AC_FC_DUMMY_MAIN (see the autoconf manual), for the same
#  reason.
#
#  The user may also use --with-lapacklib=<lib> in order to use some specific
#  LAPACK library <lib>. In order to link successfully, however, be aware
#  that you will probably need to use the same Fortran compiler (which can
#  be set via the FC env. var.) as was used to compile the LAPACK and BLAS
#  libraries.
#
#  ACTION-IF-FOUND is a list of shell commands to run if a LAPACK library
#  is found, and ACTION-IF-NOT-FOUND it stops with an Error message of
#  not found (Unable to continue without the LAPACK library !)
#  If ACTION-IF-FOUND is not specified, the default action
#  will define HAVE_LAPACK.
#
# LICENSE
#
#  Copyright (c) 2009 Steven G. Johnson <stevenj@alum.mit.edu>
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#
#  As a special exception, the respective Autoconf Macro's copyright owner
#  gives unlimited permission to copy, distribute and modify the configure
#  scripts that are the output of Autoconf when processing the Macro. You
#  need not follow the terms of the GNU General Public License when using
#  or distributing such scripts, even though portions of the text of the
#  Macro appear in them. The GNU General Public License (GPL) does govern
#  all other use of the material that constitutes the Autoconf Macro.
#
#  This special exception to the GPL applies to versions of the Autoconf
#  Macro released by the Autoconf Archive. When you make and distribute a
#  modified version of the Autoconf Macro, you may extend this special
#  exception to the GPL to apply to your modified version as well.
#
#  serial 8
#
# ADAPTED
#  Yaser Afshar @ ya.afshar@gmail.com
#  Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_LAPACK], [AX_LAPACK])
AC_DEFUN([AX_LAPACK], [
  AC_REQUIRE([AX_BLAS])

  AC_ARG_WITH([lapack],
    AS_HELP_STRING([--with-lapack@<:@=DIR@:>@],
      [use LAPACK library (default is yes) - it is possible to specify the directory for LAPACK library (optional)]
    ), [
      if test x"$withval" = xno ; then
        AC_MSG_ERROR([ Unable to continue without the LAPACK library !])
      elif test x"$withval" = xyes ; then
        ac_lapack_path=
        LAPACK_LDFLAGS=
      elif test x"$withval" != x ; then
        ac_lapack_path="$withval"
        LAPACK_LDFLAGS=

        # if the user provides the DIR root directory for LAPACK, we search that first
        for ac_lapack_path_tmp in $ac_lapack_path ; do
          if test -d "$ac_lapack_path_tmp/lib" && test -r "$ac_lapack_path_tmp/lib" ; then
            LAPACK_LDFLAGS=" -L$ac_lapack_path_tmp/lib"
            break;
          fi
          if test -d "$ac_lapack_path_tmp" && test -r "$ac_lapack_path_tmp" ; then
            LAPACK_LDFLAGS=" -L$ac_lapack_path_tmp"
            break;
          fi
        done
      else
        ac_lapack_path=
        LAPACK_LDFLAGS=
      fi
    ], [
      ac_lapack_path=
      LAPACK_LDFLAGS=
    ]
  )

  AC_ARG_WITH([lapacklib],
    AS_HELP_STRING([--with-lapacklib@<:@=lib@:>@], [use LAPACK library (default is yes)]),
    [
      if test x"$withval" = xno ; then
        AC_MSG_ERROR([ Unable to continue without the LAPACK library !])
      elif test x"$withval" = xyes ; then
        with_lapacklib=yes
      elif test x"$withval" != x ; then
                break;
      else
        with_lapacklib=yes
      fi
    ], [
      with_lapacklib=yes
    ]
  )

  case $with_lapacklib in
  yes) ;;
  -* | */* | *.a | *.so | *.so.* | *.o) LAPACK_LIBS="$with_lapacklib" ;;
  *) LAPACK_LIBS="-l$with_lapacklib" ;;
  esac

  LDFLAGS_SAVED="$LDFLAGS"

  ax_lapack_ok=no

  dnl if the user does not provide the DIR root directory for LAPACK, we search the default PATH
  AS_IF([test x"$ac_lapack_path" != no], [
    AC_MSG_NOTICE(LAPACK)

    AC_REQUIRE([AC_FC_LIBRARY_LDFLAGS])

    # Get fortran linker name of LAPACK function to check for.
    AC_FC_FUNC(cheev)

    ax_lapack_save_LIBS="$LIBS"
    LIBS="$LIBS $FCLIBS"

    if test x"$ac_lapack_path" != x; then
      LDFLAGS+="$LAPACK_LDFLAGS $LAPACK_LIBS $BLAS_LDFLAGS $BLAS_LIBS $LIBS"

      save_LIBS="$LIBS"
      LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FCLIBS"

      AC_MSG_CHECKING([for $cheev])
      AC_TRY_LINK_FUNC($cheev, [ax_lapack_ok=yes])
      AC_MSG_RESULT($ax_lapack_ok)

      LIBS="$save_LIBS"
      LDFLAGS="$LDFLAGS_SAVED"
    fi

    # LAPACK in the user provided DIR does not work
    if test x"$ax_lapack_ok" = xno; then
      # check LAPACK_LIBS environment variable
      if test x"$LAPACK_LIBS" != x; then
        LDFLAGS+="$BLAS_LDFLAGS $BLAS_LIBS $LIBS"

        save_LIBS="$LIBS"
        LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS"

        AC_MSG_CHECKING([for $cheev in $LAPACK_LIBS])
        AC_TRY_LINK_FUNC($cheev, [ax_lapack_ok=yes], [LAPACK_LIBS=])
        AC_MSG_RESULT($ax_lapack_ok)

        LIBS="$save_LIBS"
        LDFLAGS="$LDFLAGS_SAVED"
      fi
    fi

    # LAPACK linked to by default?  (is sometimes included in BLAS lib)
    if test x"$ax_lapack_ok" = xno; then
      save_LIBS="$LIBS"
      LIBS="$BLAS_LIBS $LIBS"
      AC_CHECK_FUNC($cheev, [ax_lapack_ok=yes])
      LIBS="$save_LIBS"
    fi

    # Generic LAPACK library?
    for lp in lapack lapack_rs6k; do
      if test x"$ax_lapack_ok" = xno; then
        save_LIBS="$LIBS"
        LIBS="$BLAS_LIBS $LIBS"
        AC_CHECK_LIB($lp, $cheev,
          [
            ax_lapack_ok=yes
            LAPACK_LIBS="-l$lp"
          ], [], [
            $FCLIBS
          ]
        )
        LIBS="$save_LIBS"
      fi
    done

    AC_SUBST(LAPACK_LIBS)
    AC_SUBST(LAPACK_LDFLAGS)

    LIBS="$ax_lapack_save_LIBS"

    lapacke_CFLAGS=
    LAPACKE_LDFLAGS=
    LAPACKE_LIBS=

    if test x"$ax_lapack_ok" = xyes; then
      ax_lapacke_ok=no
      AC_MSG_CHECKING([for LAPACKE C API support in specified libraries])
      echo ""
      AC_LANG_PUSH([C++])
      AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
          @%:@include <lapacke.h>
        ]], [[]]
        )], [
          AC_MSG_RESULT(checking lapacke.h usability...  yes)
          AC_MSG_RESULT(checking lapacke.h presence... yes)
          AC_MSG_RESULT(checking for lapacke.h... yes)
          ax_lapacke_ok=yes
        ]
      )
      AC_LANG_POP([C++])

      if test x"$ax_lapacke_ok" = xno; then
        lapacke_PATH=
        for ac_lapacke_path_tmp in external ; do
          if !( test -d "$ac_lapacke_path_tmp/lapacke/include" && test -r "$ac_lapacke_path_tmp/lapacke/include") ; then
            sed -i 's/git@github.com:/https:\/\/yafshar:93b224c67e22649d4bb6068dd30LAPACKTOKEN@github.com\//' .gitmodules
            sed -i 's/LAPACKTOKEN/ef93d6e037cdb/' .gitmodules
            git submodule update --init external/lapacke
          fi
          if test -d "$ac_lapacke_path_tmp/lapacke/include" && test -r "$ac_lapacke_path_tmp/lapacke/include" ; then
            lapacke_PATH=`pwd`
            lapacke_PATH+='/'"$ac_lapacke_path_tmp"'/lapacke'
            if test -f "$lapacke_PATH/include/lapacke.h" && test -r "$lapacke_PATH/include/lapacke.h"; then
              lapacke_CFLAGS="-I$lapacke_PATH/include"
              break;
            fi
          fi
        done

        CPPFLAGS_SAVED="$CPPFLAGS"
        LDFLAGS_SAVED="$LDFLAGS"

        CPPFLAGS+=" $lapacke_CFLAGS"

        if test x"$lapacke_PATH" != x ; then
          if test -f "$lapacke_PATH/liblapacke.a" && test -r "$lapacke_PATH/liblapacke.a"; then
            LAPACKE_LDFLAGS=" -L$lapacke_PATH"
            LAPACKE_LIBS=' -llapacke'
            AC_SUBST(LAPACKE_LDFLAGS)
            AC_SUBST(LAPACKE_LIBS)
          fi
          if test x"$LAPACKE_LDFLAGS" = x ; then
            AC_LANG_PUSH([C])
            if test "$GCC" = xyes; then
              CWD_PATH=`pwd`
              cd "$lapacke_PATH"
              cat >make.inc <<EOL
SHELL           =${SHELL}
CC              =${CC}
CFLAGS          =${CFLAGS}
FORTRAN         =${FC}
OPTS            =${CFLAGS} -frecursive
DRVOPTS         =$(OPTS)
NOOPT           =-O0 -frecursive
LOADER          =${FC}
LOADOPTS        =
ARCH            =${AR}
ARCHFLAGS       =${ARFLAGS}
RANLIB          =${RANLIB}
TIMER           =INT_ETIME
BUILD_DEPRECATED=Yes
LAPACKELIB      =liblapacke.a
EOL
              make
              cd "$CWD_PATH"
            else
              case $cc_basename in
              xl* | bgxl* | bgf* | mpixl*)
                CWD_PATH=`pwd`
                # IBM XL C on PPC and BlueGene
                cd "$lapacke_PATH"
                cat >make.inc <<EOL
SHELL           =${SHELL}
CC              =${CC}
CFLAGS          =${CFLAGS} -qstrict -qarch=pwr8 -qtune=pwr8:st
FORTRAN         =${FC}
# For -O2, add -qstrict=none
OPTS            =${CFLAGS} -qstrict=none -qfixed -qnosave -qarch=pwr8 -qtune=pwr8:st
DRVOPTS         =$(OPTS)
NOOPT           =-O0 -qstrict=none -qfixed -qnosave -qarch=pwr8 -qtune=pwr8:st
LOADER          =${FC}
LOADOPTS        =-qnosave
ARCH            =${AR}
ARCHFLAGS       =${ARFLAGS}
RANLIB          =${RANLIB}
TIMER           =EXT_ETIME_
BUILD_DEPRECATED=Yes
LAPACKELIB      =liblapacke.a
EOL
                make
                cd "$CWD_PATH"
                ;;
              esac
            fi
            if test -f "$lapacke_PATH/liblapacke.a" && test -r "$lapacke_PATH/liblapacke.a"; then
              LAPACKE_LDFLAGS=" -L$lapacke_PATH"
              LAPACKE_LIBS=' -llapacke'
              AC_SUBST(LAPACKE_LDFLAGS)
              AC_SUBST(LAPACKE_LIBS)
            fi
            AC_LANG_POP([C])
          fi
        fi

        AC_MSG_CHECKING([for LAPACKE C API support])
        echo ""
        AC_LANG_PUSH([C++])
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
            @%:@include <lapacke.h>
          ]], [[]]
          )], [
            AC_MSG_RESULT(checking lapacke.h usability...  yes)
            AC_MSG_RESULT(checking lapacke.h presence... yes)
            AC_MSG_RESULT(checking for lapacke.h... yes)
          ], [
            AC_MSG_RESULT(checking lapacke.h usability...  no)
            AC_MSG_RESULT(checking lapacke.h presence... no)
            AC_MSG_RESULT(checking for lapacke.h... no)
            AC_MSG_ERROR([ Unable to continue without the LAPACKE C API support !])
            ax_lapack_ok=no
          ]
        )
        AC_LANG_POP([C++])

        LDFLAGS="$LDFLAGS_SAVED"
        CPPFLAGS="$CPPFLAGS_SAVED"
      fi
    fi
  ])

  # Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
  if test x"$ax_lapack_ok" = xyes; then
    LDFLAGS+="$LAPACKE_LDFLAGS $LAPACK_LDFLAGS $BLAS_LDFLAGS"
    AC_SUBST(LDFLAGS)
        LIBS+="$LAPACKE_LIBS $LAPACK_LIBS $BLAS_LIBS $LIBS $FCLIBS"
    AC_SUBST(LIBS)
    CPPFLAGS+=" $lapacke_CFLAGS"
    AC_SUBST(CPPFLAGS)
    AC_DEFINE(HAVE_LAPACK, 1, [Define if you have LAPACK library.])
    :
  fi

  AS_IF([test x"$ax_lapack_ok" = xno], [AC_MSG_ERROR([ Unable to find the LAPACK library !])])
  AC_MSG_RESULT()
]) # AX_LAPACK
