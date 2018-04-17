# ===========================================================================
#        https://www.gnu.org/software/autoconf-archive/ax_lapack.html
# ===========================================================================
#
# SYNOPSIS
#
#	AX_LAPACK([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#	This macro looks for a library that implements the LAPACK linear-algebra
#	interface (see http://www.netlib.org/lapack/). On success, it sets the
#	LAPACK_LIBS output variable to hold the requisite library linkages.
#	
#	To link with LAPACK, you should link with:
#	
#	$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS
#	
#	in that order. BLAS_LIBS is the output variable of the AX_BLAS macro,
#	called automatically. FLIBS is the output variable of the
#	AC_F77_LIBRARY_LDFLAGS macro (called if necessary by AX_BLAS), and is
#	sometimes necessary in order to link with F77 libraries. Users will also
#	need to use AC_F77_DUMMY_MAIN (see the autoconf manual), for the same
#	reason.
#	
#	The user may also use --with-lapack=<lib> in order to use some specific
#	LAPACK library <lib>. In order to link successfully, however, be aware
#	that you will probably need to use the same Fortran compiler (which can
#	be set via the F77 env. var.) as was used to compile the LAPACK and BLAS
#	libraries.
#	
#	ACTION-IF-FOUND is a list of shell commands to run if a LAPACK library
#	is found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it
#	is not found. If ACTION-IF-FOUND is not specified, the default action
#	will define HAVE_LAPACK.
#
# LICENSE
#
#	Copyright (c) 2009 Steven G. Johnson <stevenj@alum.mit.edu>
#	
#	This program is free software: you can redistribute it and/or modify it
#	under the terms of the GNU General Public License as published by the
#	Free Software Foundation, either version 3 of the License, or (at your
#	option) any later version.
#	
#	This program is distributed in the hope that it will be useful, but
#	WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#	Public License for more details.
#	
#	You should have received a copy of the GNU General Public License along
#	with this program. If not, see <https://www.gnu.org/licenses/>.
#	
#	As a special exception, the respective Autoconf Macro's copyright owner
#	gives unlimited permission to copy, distribute and modify the configure
#	scripts that are the output of Autoconf when processing the Macro. You
#	need not follow the terms of the GNU General Public License when using
#	or distributing such scripts, even though portions of the text of the
#	Macro appear in them. The GNU General Public License (GPL) does govern
#	all other use of the material that constitutes the Autoconf Macro.
#	
#	This special exception to the GPL applies to versions of the Autoconf
#	Macro released by the Autoconf Archive. When you make and distribute a
#	modified version of the Autoconf Macro, you may extend this special
#	exception to the GPL to apply to your modified version as well.
#
#	serial 8
#
# ADAPTED 
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan

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
			elif test x"$withval" != x ; then
				ac_lapack_path="$withval"
			else
				ac_lapack_path=
			fi
		], [
			ac_lapack_path=
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
                with_lapacklib="$withval"
			else
				with_lapacklib=yes
			fi
		], [
			with_lapacklib=yes
		]
	)

	LDFLAGS_SAVED="$LDFLAGS"
	lapack_LDFLAGS=

	case $with_lapacklib in
	yes) ;;
	-* | */* | *.a | *.so | *.so.* | *.o) LAPACK_LIBS="$with_lapacklib" ;;
	*) LAPACK_LIBS="-l$with_lapacklib" ;;
    esac

	ax_lapack_ok=no

	dnl if the user does not provide the DIR root directory for LAPACK, we search the default PATH
	AS_IF([test x"$ac_lapack_path" = no], [], [ 
		AC_MSG_NOTICE(LAPACK)

		# Get fortran linker name of LAPACK function to check for.
		AC_F77_FUNC(cheev)

		# We cannot use LAPACK if BLAS is not found
		if test x"$ax_blas_ok" != xyes; then
			ax_lapack_ok=noblas
			LAPACK_LIBS=
		fi

		AS_IF([test x"$ac_lapack_path" = x], [
		    # First, check LAPACK_LIBS environment variable
			if test x"$LAPACK_LIBS" != x; then
				save_LIBS="$LIBS"; 
				LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"
                
				AC_MSG_CHECKING([for $cheev in $LAPACK_LIBS])
				AC_TRY_LINK_FUNC($cheev, 
					[ax_lapack_ok=yes], [LAPACK_LIBS=]
				)
				AC_MSG_RESULT($ax_lapack_ok)
                
				LIBS="$save_LIBS"
				if test x"$ax_lapack_ok" = xno; then
					LAPACK_LIBS=
				fi
			fi
		], [
            # if the user provides the DIR root directory for LAPACK, we search that first
			for ac_lapack_path_tmp in $ac_lapack_path ; do
				if test -d "$ac_lapack_path_tmp/lib" && test -r "$ac_lapack_path_tmp/lib" ; then
					lapack_LDFLAGS+=" -L$ac_lapack_path_tmp/lib"
					break;
				fi
				if test -d "$ac_lapack_path_tmp" && test -r "$ac_lapack_path_tmp" ; then
					lapack_LDFLAGS+=" -L$ac_lapack_path_tmp"
					break;
				fi
			done        
            
            LDFLAGS+=" $lapack_LDFLAGS $LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"
				
			save_LIBS="$LIBS"; 
			LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"
				
			AC_MSG_CHECKING([for $cheev])
			AC_TRY_LINK_FUNC($cheev, [
					ax_lapack_ok=yes
					AC_SUBST(LDFLAGS)
					AC_SUBST(LIBS)
				], [
					LDFLAGS="$LDFLAGS_SAVED"
					LIBS="$save_LIBS"
				]
			)
			AC_MSG_RESULT($ax_lapack_ok)

            # LAPACK in the user provided DIR does not work
            if test x"$ax_lapack_ok" = xno; then
                # check LAPACK_LIBS environment variable
			    if test x"$LAPACK_LIBS" != x; then
				    LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"
                
				    AC_MSG_CHECKING([for $cheev in $LAPACK_LIBS])
				    AC_TRY_LINK_FUNC($cheev, 
					    [ax_lapack_ok=yes], [LAPACK_LIBS=]
				    )
				    AC_MSG_RESULT($ax_lapack_ok)
                
				    LIBS="$save_LIBS"
				    if test x"$ax_lapack_ok" = xno; then
					    LAPACK_LIBS=
				    fi
                else
                    LAPACK_LIBS=
			    fi
            fi
		])

		# LAPACK linked to by default?  (is sometimes included in BLAS lib)
		if test x"$ax_lapack_ok" = xno; then
			save_LIBS="$LIBS"; 
			LIBS="$LIBS $BLAS_LIBS $FLIBS"
			AC_CHECK_FUNC($cheev, [ax_lapack_ok=yes])
			LIBS="$save_LIBS"
		fi

		# Generic LAPACK library?
		for lapack in lapack lapack_rs6k; do
			if test x"$ax_lapack_ok" = xno; then
				save_LIBS="$LIBS"; 
				LIBS="$BLAS_LIBS $LIBS"
				AC_CHECK_LIB($lapack, $cheev,
					[
						ax_lapack_ok=yes
						LAPACK_LIBS="-l$lapack"
					], [], [
						$FLIBS
					]
				)
				LIBS="$save_LIBS"
			fi
		done

		AC_SUBST(LAPACK_LIBS)

		AC_LANG_PUSH([C++])
		if test x"$ax_lapack_ok" = xyes; then
			AC_CHECK_LIB(lapack, dgetrf_, 
				[], [
					ax_lapack_ok=no
					AC_MSG_ERROR([ Unable to continue without the LAPACK library !])
				]
			)
		fi  
		AC_LANG_POP([C++])
	])

	if test x"$ax_lapack_ok" = xyes; then
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
			], [
				AC_MSG_RESULT(checking lapacke.h usability...  no)
				AC_MSG_RESULT(checking lapacke.h presence... no)
				AC_MSG_RESULT(checking for lapacke.h... no)
				AC_MSG_ERROR([ Unable to continue without the LAPACKE C API support !])
				ax_lapack_ok=no
			]
		)
		AC_LANG_POP([C++])
	fi

	# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
	if test x"$ax_lapack_ok" = xyes; then
		LDFLAGS+=" $LAPACK_LIBS"
		AC_SUBST(LDFLAGS)
		AC_DEFINE(HAVE_LAPACK, 1, [Define if you have LAPACK library.])
		:
	fi

	AS_IF([test x"$ax_lapack_ok" = xno], [AC_MSG_ERROR([ Unable to find the LAPACK library !])])
	AC_MSG_RESULT()
]) # AX_LAPACK
