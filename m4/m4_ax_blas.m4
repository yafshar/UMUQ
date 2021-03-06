# ===========================================================================
#         https://www.gnu.org/software/autoconf-archive/ax_blas.html
# ===========================================================================
#
# SYNOPSIS
#
#  AX_BLAS([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#  This macro looks for a library that implements the BLAS linear-algebra
#  interface (see http://www.netlib.org/blas/). On success, it sets the
#  BLAS_LIBS output variable to hold the requisite library linkages.
#
#  To link with BLAS, you should link with:
#
#  $BLAS_LIBS $LIBS $FCLIBS
#
#  in that order. FCLIBS is the output variable of the
#  AC_FC_LIBRARY_LDFLAGS macro (called if necessary by AX_BLAS), and is
#  sometimes necessary in order to link with FC libraries. Users will also
#  need to use AC_FC_DUMMY_MAIN (see the autoconf manual), for the same
#  reason.
#
#  Many libraries are searched for, from ATLAS to CXML to ESSL. The user
#  may also use --with-blaslib=<lib> in order to use some specific BLAS
#  library <lib>. In order to link successfully, however, be aware that you
#  will probably need to use the same Fortran compiler (which can be set
#  via the FC env. var.) as was used to compile the BLAS library.
#
#  ACTION-IF-FOUND is a list of shell commands to run if a BLAS library is
#  found, and ACTION-IF-NOT-FOUND it stops with an Error message of
#  not found (Unable to continue without the BLAS library !)
#  If ACTION-IF-FOUND is not specified, the default action will
#  define HAVE_BLAS.
#
# LICENSE
#
#  Copyright (c) 2008 Steven G. Johnson <stevenj@alum.mit.edu>
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
#  serial 15
#
# ADAPTED
#  Yaser Afshar @ ya.afshar@gmail.com
#  Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_BLAS], [AX_BLAS])
AC_DEFUN([AX_BLAS], [
  AC_ARG_WITH([blas],
    AS_HELP_STRING([--with-blas@<:@=DIR@:>@],
      [use BLAS library (default is yes) - it is possible to specify the directory for BLAS library (optional)]
    ), [
      if test x"$withval" = xno ; then
        AC_MSG_ERROR([ Unable to continue without the BLAS library !])
      elif test x"$withval" = xyes ; then
        ac_blas_path=
        BLAS_LDFLAGS=
      elif test x"$withval" != x ; then
        ac_blas_path="$withval"
        BLAS_LDFLAGS=

        # if the user provides the DIR root directory for BLAS, we check that first
        for ac_blas_path_tmp in $ac_blas_path ; do
          if test -d "$ac_blas_path_tmp/lib" && test -r "$ac_blas_path_tmp/lib" ; then
            BLAS_LDFLAGS=" -L$ac_blas_path_tmp/lib"
            break;
          fi
          if test -d "$ac_blas_path_tmp" && test -r "$ac_blas_path_tmp" ; then
            BLAS_LDFLAGS=" -L$ac_blas_path_tmp"
            break;
          fi
        done
      else
        ac_blas_path=
        BLAS_LDFLAGS=
      fi
    ], [
      ac_blas_path=
      BLAS_LDFLAGS=
    ]
  )

  AC_ARG_WITH([blaslib],
    AS_HELP_STRING([--with-blaslib@<:@=lib@:>@], [use BLAS library (default is yes)]),
    [
      if test x"$withval" = xno ; then
        AC_MSG_ERROR([ Unable to continue without the BLAS library !])
      elif test x"$withval" = xyes ; then
        break;
      elif test x"$withval" != x ; then
        break;
      else
        with_blaslib=yes
      fi
    ], [
       with_blaslib=yes
    ]
  )

  case $with_blaslib in
  yes) ;;
  -* | */* | *.a | *.so | *.so.* | *.o) BLAS_LIBS="$with_blaslib" ;;
  *) BLAS_LIBS="-l$with_blaslib" ;;
  esac

  LDFLAGS_SAVED="$LDFLAGS"

  ax_blas_ok=no

  # if the user does not provide the DIR root directory for BLAS, we search the default PATH
  AS_IF([test x"$ac_blas_path" != xno], [
    AC_MSG_NOTICE(BLAS)

    AC_REQUIRE([AC_FC_LIBRARY_LDFLAGS])
    AC_REQUIRE([AC_CANONICAL_HOST])

    # Get fortran linker names of BLAS functions to check for.
    AC_FC_FUNC(sgemm)
    AC_FC_FUNC(dgemm)

    ax_blas_save_LIBS="$LIBS"
    LIBS="$LIBS $FCLIBS"

    # First check BLAS_PATH & BLAS_LIBS environment variables
    AS_IF([test x"$ac_blas_path" != x], [
      LDFLAGS+="$BLAS_LDFLAGS $BLAS_LIBS $LIBS"

      save_LIBS="$LIBS";
      LIBS="$BLAS_LIBS $LIBS"

      AC_MSG_CHECKING([for $sgemm])
      AC_TRY_LINK_FUNC($sgemm, [ax_blas_ok=yes])
      AC_MSG_RESULT($ax_blas_ok)

      LIBS="$save_LIBS"
      LDFLAGS="$LDFLAGS_SAVED"
    ])

    # Check BLAS_LIBS
    if test x"$ax_blas_ok" = xno; then
      if test x"$BLAS_LIBS" != x; then
        save_LIBS="$LIBS";
        LIBS="$BLAS_LIBS $LIBS"

        AC_MSG_CHECKING([for $sgemm in $BLAS_LIBS])
        AC_TRY_LINK_FUNC($sgemm, [ax_blas_ok=yes], [BLAS_LIBS=])
        AC_MSG_RESULT($ax_blas_ok)

        LIBS="$save_LIBS"
      fi
    fi

    # BLAS linked to by default?  (happens on some supercomputers)
    if test x"$ax_blas_ok" = xno; then
      save_LIBS="$LIBS";
      LIBS="$LIBS"

      AC_MSG_CHECKING([if $sgemm is being linked in already])
      AC_TRY_LINK_FUNC($sgemm, [ax_blas_ok=yes])
      AC_MSG_RESULT($ax_blas_ok)

      LIBS="$save_LIBS"
    fi

    # BLAS in OpenBLAS library? (http://xianyi.github.com/OpenBLAS/)
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(openblas, $sgemm,
        [
          ax_blas_ok=yes
          BLAS_LIBS="-lopenblas"
        ]
      )
    fi

    # BLAS in ATLAS library? (http://math-atlas.sourceforge.net/)
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(atlas, ATL_xerbla,
        [
          AC_CHECK_LIB(f77blas, $sgemm,
            [
              AC_CHECK_LIB(cblas, cblas_dgemm,
                [
                  ax_blas_ok=yes
                  BLAS_LIBS="-lcblas -lf77blas -latlas"
                ], [], [
                  -lf77blas -latlas
                ]
              )
            ], [], [
              -latlas
            ]
          )
        ]
      )
    fi

    # BLAS in PhiPACK libraries? (requires generic BLAS lib, too)
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(blas, $sgemm,
        [
          AC_CHECK_LIB(dgemm, $dgemm,
            [
              AC_CHECK_LIB(sgemm, $sgemm,
                [
                  ax_blas_ok=yes
                  BLAS_LIBS="-lsgemm -ldgemm -lblas"
                ], [], [
                  -lblas
                ]
              )
            ], [], [
              -lblas
            ]
          )
        ]
      )
    fi

    # BLAS in Intel MKL library?
    if test x"$ax_blas_ok" = xno; then
      # MKL for gfortran
      if test x"$ac_cv_fc_compiler_gnu" = xyes; then
        # 64 bit
        if test x"$host_cpu" = x86_64; then
          AC_CHECK_LIB(mkl_gf_lp64, $sgemm,
            [
              ax_blas_ok=yes
              BLAS_LIBS='-lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread'
            ], , [
              -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread
            ]
          )
        # 32 bit
        elif test x"$host_cpu" = xi686; then
          AC_CHECK_LIB(mkl_gf, $sgemm,
            [
              ax_blas_ok=yes
              BLAS_LIBS='-lmkl_gf -lmkl_sequential -lmkl_core -lpthread'
            ], , [
              -lmkl_gf -lmkl_sequential -lmkl_core -lpthread
            ]
          )
        fi
      # MKL for other compilers (Intel, PGI, ...?)
      else
        # 64-bit
        if test x"$host_cpu" = x86_64; then
          AC_CHECK_LIB(mkl_intel_lp64, $sgemm,
            [
              ax_blas_ok=yes
              BLAS_LIBS='-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread'
            ], , [
              -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread
            ]
          )
        # 32-bit
        elif test x"$host_cpu" = xi686; then
          AC_CHECK_LIB(mkl_intel, $sgemm,
            [
              ax_blas_ok=yes
              BLAS_LIBS='-lmkl_intel -lmkl_sequential -lmkl_core -lpthread'
            ], , [
              -lmkl_intel -lmkl_sequential -lmkl_core -lpthread
            ]
          )
        fi
      fi
    fi

    # Old versions of MKL
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(mkl, $sgemm,
        [
          ax_blas_ok=yes
          BLAS_LIBS='-lmkl -lguide -lpthread'
        ], , [
          -lguide -lpthread
        ]
      )
    fi

    # BLAS in Apple vecLib library?
    if test x"$ax_blas_ok" = xno; then
      save_LIBS="$LIBS"
      LIBS="-framework vecLib $LIBS"

      AC_MSG_CHECKING([for $sgemm in -framework vecLib])
      AC_TRY_LINK_FUNC($sgemm,
        [
          ax_blas_ok=yes
          BLAS_LIBS="-framework vecLib"
        ]
      )
      AC_MSG_RESULT($ax_blas_ok)

      LIBS="$save_LIBS"
    fi

    # BLAS in Alpha CXML library?
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(cxml, $sgemm,
        [
          ax_blas_ok=yes
          BLAS_LIBS="-lcxml"
        ]
      )
    fi

    # BLAS in Alpha DXML library? (now called CXML, see above)
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(dxml, $sgemm,
        [
          ax_blas_ok=yes
          BLAS_LIBS="-ldxml"
        ]
      )
    fi

    # BLAS in Sun Performance library?
    if test x"$ax_blas_ok" = xno; then
      # only works with Sun CC
      if test x"$GCC" != xyes; then
        AC_CHECK_LIB(sunmath, acosp,
          [
            AC_CHECK_LIB(sunperf, $sgemm,
              [
                BLAS_LIBS='-xlic_lib=sunperf -lsunmath'
                ax_blas_ok=yes
              ], [], [
                -lsunmath
              ]
            )
          ]
        )
      fi
    fi

    # BLAS in SCSL library?  (SGI/Cray Scientific Library)
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(scs, $sgemm,
        [
          ax_blas_ok=yes
          BLAS_LIBS='-lscs'
        ]
      )
    fi

    # BLAS in SGIMATH library?
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(complib.sgimath, $sgemm,
        [
          ax_blas_ok=yes
          BLAS_LIBS='-lcomplib.sgimath'
        ]
      )
    fi

    # BLAS in IBM ESSL library? (requires generic BLAS lib, too)
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(blas, $sgemm,
        [
          AC_CHECK_LIB(essl, $sgemm,
            [
              ax_blas_ok=yes
              BLAS_LIBS='-lessl -lblas'
            ], [], [
              -lblas $FCLIBS
            ]
          )
        ]
      )
    fi

    # Generic BLAS library?
    if test x"$ax_blas_ok" = xno; then
      AC_CHECK_LIB(blas, $sgemm,
        [
          ax_blas_ok=yes
          BLAS_LIBS='-lblas'
        ]
      )
    fi

    AC_SUBST(BLAS_LIBS)
    AC_SUBST(BLAS_LDFLAGS)

    LIBS="$ax_blas_save_LIBS"
  ])

  # Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
  if test x"$ax_blas_ok" = xyes; then
    AC_DEFINE(HAVE_BLAS, 1, [Define if you have a BLAS library.])
    :
  fi

  AS_IF([test x"$ax_blas_ok" = xno], [ AC_MSG_ERROR([ Unable to find the BLAS library !])])
  AC_MSG_RESULT()
]) # AX_BLAS
