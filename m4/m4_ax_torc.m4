#
# SYNOPSIS
#
#   AX_TORC([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   Test for the TORC libraries
#
#   If no path to the installed TORC library is given the macro searchs
#   under /usr, /usr/local, /usr/local/include, /opt, and /opt/local 
#   and evaluates the environment variable for TORC library and header files. 
#
# AUTHOR 
#  Yaser Afshar @ ya.afshar@gmail.com
#  Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_TORC], [AX_TORC])
AC_DEFUN([AX_TORC], [
    AC_ARG_WITH([torc], 
		AS_HELP_STRING([--with-torc@<:@=DIR@:>@], 
			[use TORC library (default is yes) - it is possible to specify the root directory for TORC library (optional)]),            
		[ 
			if test x"$withval" = xno; then
				AC_MSG_ERROR([ Unable to continue without the TORC library !])
			elif test x"$withval" = xyes; then
                ac_torc_path=""
			elif test x"$withval" != x; then
				ac_torc_path="$withval"
			else
				ac_torc_path=""
			fi
		], [ac_torc_path=""]
    )

    dnl if the user does not provide the DIR root directory for FLANN, we search the default PATH
    AS_IF([test x"$ac_torc_path" = x], [ 
        AC_CHECK_HEADERS([torc.h], [ax_torc_ok=yes], [ax_torc_ok=no])
        
		AS_IF([test x"$ax_torc_ok" = xyes], [
            LDFLAGS_SAVED="$LDFLAGS"
            LDFLAGS+=' -ltorc '" $PTHREAD_LIBS "
            AC_LANG_PUSH(C)
            AC_CHECK_LIB(torc, torc_init, [], 
            	[
                    ax_torc_ok="no"
                    LDFLAGS="$LDFLAGS_SAVED"
                ]
            )
        	AC_LANG_POP([C])
            AC_SUBST(LDFLAGS)
        ])
        ], [
            ax_torc_ok="no"
        ]
	) 

    AS_IF([test x"$ax_torc_ok" = xno], [  
		AC_MSG_NOTICE(TORC)
                
        succeeded=no
                
		TORC_CFLAGS=""

		dnl first we check the system location for TORC libraries
		if test x"$ac_torc_path" != x; then
        	for ac_torc_path_tmp in $ac_torc_path $ac_torc_path/include $ac_torc_path/torc $ac_torc_path/torc/include ; do 
				if test -d "$ac_torc_path_tmp" && test -r "$ac_torc_path_tmp" ; then
					if test -f "$ac_torc_path_tmp/torc.h" && test -r "$ac_torc_path_tmp/torc.h" ; then
						TORC_CFLAGS="-I$ac_torc_path_tmp"" $PTHREAD_CFLAGS "
						break;
					fi
				fi
			done
		else                   
			for ac_torc_path_tmp in /usr /usr/local /usr/include /use/local/include /opt /opt/local /usr/torc /usr/local/torc /usr/include/torc /use/local/include/torc /opt/torc /opt/local/torc ; do
				if test -d "$ac_torc_path_tmp" && test -r "$ac_torc_path_tmp" ; then
					if test -f "$ac_torc_path_tmp/torc.h" && test -r "$ac_torc_path_tmp/torc.h"; then
						TORC_CFLAGS="-I$ac_torc_path_tmp/torc"" $PTHREAD_CFLAGS "
						break;
					fi
					if test -d "$ac_torc_path_tmp/include" && test -r "$ac_torc_path_tmp/include" ; then
						if test -f "$ac_torc_path_tmp/include/torc.h" && test -r "$ac_torc_path_tmp/include/torc.h"; then
							TORC_CFLAGS="-I$ac_torc_path_tmp/include"" $PTHREAD_CFLAGS "
							break;
						fi
					fi
				fi	
			done
		fi

		CFLAGS_SAVED="$CFLAGS"
		LDFLAGS_SAVED="$LDFLAGS"
		
		CFLAGS+=" $TORC_CFLAGS"
		TORC_LDFLAGS=""

		if test x"$ac_torc_path" != x; then
			for ac_torc_path_tmp in $ac_torc_path $ac_torc_path/lib $ac_torc_path/torc $ac_torc_path/torc/lib ; do
				if test -d "$ac_torc_path_tmp" && test -r "$ac_torc_path_tmp" ; then
					if test -f "$ac_torc_path_tmp/libtorc.a" && test -r "$ac_torc_path_tmp/libtorc.a"; then
						TORC_LDFLAGS="-L$ac_torc_path_tmp" 
						break;
					fi
					if test -f "$ac_torc_path_tmp/libtorc.dylib" && test -r "$ac_torc_path_tmp/libtorc.dylib"; then
						TORC_LDFLAGS="-L$ac_torc_path_tmp" 
						break;
					fi
				fi
			done
		else
			for ac_torc_path_tmp in /usr/lib /usr/lib64 /use/local/lib /use/local/lib64 /opt /opt/lib /usr/lib/torc /usr/lib64/torc /use/local/lib/torc /use/local/lib64/torc /opt/torc /opt/lib/torc /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib/arm-linux-gnueabihf /usr/lib/i386-linux-gnu /usr/lib/powerpc-linux-gnu /usr/lib/powerpc64le-linux-gnu ; do
				if test -d "$ac_torc_path_tmp" && test -r "$ac_torc_path_tmp" ; then
					if test -f "$ac_torc_path_tmp/libtorc.a" && test -r "$ac_torc_path_tmp/libtorc.a"; then
						TORC_LDFLAGS="-L$ac_torc_path_tmp" 
						break;
					fi      
					if test -f "$ac_torc_path_tmp/libtorc.dylib" && test -r "$ac_torc_path_tmp/libtorc.dylib"; then
						TORC_LDFLAGS="-L$ac_torc_path_tmp" 
						break;
					fi
				fi
			done
		fi

		LDFLAGS+=" $TORC_LDFLAGS"' -ltorc '" $PTHREAD_LIBS "         

		save_CC="$CC"
		AS_IF([test x"$ax_mpi_ok" = xyes], [CC="$MPICC"])
		
		AC_LANG_PUSH(C)
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
				@%:@include <torc.h>
			]], [[]]
			)], [
				AC_MSG_RESULT(checking torc.h usability...  yes)
				AC_MSG_RESULT(checking torc.h presence... yes)
				AC_MSG_RESULT(checking for torc.h... yes)
				succeeded=yes
			],[
				AC_MSG_RESULT(checking torc.h usability...  no)
				AC_MSG_RESULT(checking torc.h presence... no)
				AC_MSG_RESULT(checking for torc.h... no)
				AC_MSG_ERROR([ Unable to continue without the TORC header files !])
			]
		)

		AC_CHECK_LIB(torc, torc_init, 
			[], [
				succeeded=no
				AC_MSG_ERROR([ Unable to continue without the TORC library !])
			]
		)
		AC_LANG_POP([C])

		CC="$save_CC"

		if test x"$succeeded" = xyes ; then
			AC_SUBST(CFLAGS)
			AC_SUBST(LDFLAGS)
			ax_torc_ok="yes"
		else
			CFLAGS="$CFLAGS_SAVED"
			LDFLAGS="$LDFLAGS_SAVED"
		fi
    
	])
    
	AS_IF([test x"$ax_torc_ok" = xno], [ AC_MSG_ERROR([ Unable to find the TORC library !])])
    AC_MSG_RESULT()
])
