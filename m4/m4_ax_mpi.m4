#
# SYNOPSIS
#
#   AX_MPI([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro tries to find out how to compile programs that use MPI
#   (Message Passing Interface), a standard API for parallel process
#   communication (see http://www-unix.mcs.anl.gov/mpi/)
#
#   On success, it sets the MPICC, MPICXX, or MPIF77 output variable to
#   the name of the MPI compiler, depending upon the current language.
#   (This may just be $CC/$CXX/$F77, but is more often something like
#   mpicc/mpiCC/mpif77.) It also sets MPILIBS to any libraries that are
#   needed for linking MPI (e.g. -lmpi, if a special
#   MPICC/MPICXX/MPIF77 was not found).
#   
#   NOTE: 
#   If you want to compile everything with MPI, you should set:
#
#       CC="$MPICC" #OR# CXX="$MPICXX" #OR# F77="$MPIF77" #OR# FC="$MPIFC"
#       LIBS="$MPILIBS $LIBS"
#   
#   The user can force a particular library/compiler by setting the
#   MPICC/MPICXX/MPIF77/MPIFC and/or MPILIBS environment variables.
#
#   ACTION-IF-FOUND is a list of shell commands to run if an MPI
#   library is found, and ACTION-IF-NOT-FOUND is a list of commands to
#   run it if it is not found. If ACTION-IF-FOUND is not specified, the
#   default action will define HAVE_MPI.
#
#   @category InstalledPackages
#   @author Steven G. Johnson <stevenj@alum.mit.edu>
#   @version 2004-11-05
#   @license GPLWithACException
#
# ADAPTED 
#   Yaser Afshar @ ya.afshar@gmail.com

AU_ALIAS([ACX_MPI], [AX_MPI])
AC_DEFUN([AX_MPI], [
	AC_MSG_NOTICE()

	ax_mpi_ok="no"
	want_mpi="yes"
	ac_mpi_path=""

	AC_ARG_WITH([mpi], 
		AS_HELP_STRING([--with-mpi@<:@=DIR@:>@], 
			[use MPI library (default is yes) - it is possible to specify the root directory for MPI (optional)]),            
		[ 
			if test x"$withval" = xno ; then
                AC_MSG_ERROR([ Unable to continue without the MPI library !])
            elif test x"$withval" = xyes ; then
                want_mpi="yes"
                ac_mpi_path=""
            else
                want_mpi="yes"
                ac_mpi_path="$withval"
            fi
        ], [want_mpi="yes"]
    )

	if test x"$want_mpi" = xyes; then
		ac_mpi_bin=""
    
	    if test x"$ac_mpi_path" != x; then
			for ac_mpi_path_tmp in $ac_mpi_path $ac_mpi_path/bin ; do
				for ax_mpi_tmp_CC in mpicc hcc mpcc mpcc_r mpxlc cmpicc ; do
					if test -x "$ac_mpi_path_tmp"'/'"$ax_mpi_tmp_CC" ; then
						ac_mpi_bin = ac_mpi_path_tmp
						break;
					fi
				done
			done
		fi
		
		AC_PREREQ(2.50) 
	    AC_LANG_CASE(
	        [C], [
	            AC_REQUIRE([AC_PROG_CC])
	            AC_ARG_VAR(MPICC, [MPI C compiler command])
	            if test x"$ac_mpi_bin" = x; then
					AC_CHECK_PROGS(MPICC, mpicc hcc mpcc mpcc_r mpxlc cmpicc, $CC)
				else
					AC_CHECK_PROGS(MPICC, "$ac_mpi_bin"'/mpicc' "$ac_mpi_bin"'/hcc' "$ac_mpi_bin"'/mpcc' "$ac_mpi_bin"'/mpcc_r' "$ac_mpi_bin"'/mpxlc' "$ac_mpi_bin"'/cmpicc', $CC)
				fi
	            ax_mpi_save_CC="$CC"
	            LAMMPICC="$CC"
	            CC="$MPICC"
	            AC_SUBST(MPICC)
	        ],
	        [C++], [
	            AC_REQUIRE([AC_PROG_CXX])
	            AC_ARG_VAR(MPICXX, [MPI C++ compiler command])
	            if test x"$ac_mpi_bin" = x; then
	            	AC_CHECK_PROGS(MPICXX, mpic++ mpicxx mpiCC mpCC hcp mpxlC mpxlC_r cmpic++, $CXX)
				else
					AC_CHECK_PROGS(MPICC, "$ac_mpi_bin"'/mpic++' "$ac_mpi_bin"'/mpicxx' "$ac_mpi_bin"'/mpiCC' "$ac_mpi_bin"'/mpCC' "$ac_mpi_bin"'/hcp' "$ac_mpi_bin"'/mpxlC' "$ac_mpi_bin"'/mpxlC_r' "$ac_mpi_bin"'/cmpic++', $CXX)
				fi
	            ax_mpi_save_CXX="$CXX"
	            LAMMPICXX="$CXX"
	            CXX="$MPICXX"
	            AC_SUBST(MPICXX)
	        ],
	        [Fortran 77], [
	            AC_REQUIRE([AC_PROG_F77])
	            AC_ARG_VAR(MPIF77, [MPI Fortran compiler command])
	            if test x"$ac_mpi_bin" = x; then
	            	AC_CHECK_PROGS(MPIF77, mpifort mpif77 hf77 mpxlf mpf77 mpif90 mpf90 mpxlf90 mpxlf95 mpxlf_r cmpifc cmpif90c, $F77)
				else
					AC_CHECK_PROGS(MPICC, "$ac_mpi_bin"'/mpifort' "$ac_mpi_bin"'/mpif77' "$ac_mpi_bin"'/hf77' "$ac_mpi_bin"'/mpxlf' "$ac_mpi_bin"'/mpxlf' "$ac_mpi_bin"'/mpif90' "$ac_mpi_bin"'/mpf90' "$ac_mpi_bin"'/mpxlf90' "$ac_mpi_bin"'/mpxlf95' "$ac_mpi_bin"'/mpxlf_r' "$ac_mpi_bin"'/cmpifc' "$ac_mpi_bin"'/cmpif90c', $F77)
				fi
	            ax_mpi_save_F77="$F77"
	            LAMMPIF77="$F77"
	            F77="$MPIF77"
	            AC_SUBST(MPIF77)
	        ],
	        [Fortran], [
		        AC_REQUIRE([AC_PROG_FC])
		        AC_ARG_VAR(MPIFC, [MPI Fortran compiler command])
	            if test x"$ac_mpi_bin" = x; then
		        	AC_CHECK_PROGS(MPIFC, mpifort mpif90 mpxlf95_r mpxlf90_r mpxlf95 mpxlf90 mpf90 cmpif90c, $FC)
				else
					AC_CHECK_PROGS(MPICC, "$ac_mpi_bin"'/mpifort' "$ac_mpi_bin"'/mpif90' "$ac_mpi_bin"'/mpxlf95_r' "$ac_mpi_bin"'/mpxlf90_r' "$ac_mpi_bin"'/mpxlf95' "$ac_mpi_bin"'/mpxlf90' "$ac_mpi_bin"'/mpf90' "$ac_mpi_bin"'/mpf90' "$ac_mpi_bin"'/cmpif90c', $FC)
				fi
		        ax_mpi_save_FC="$FC"
		        FC="$MPIFC"
	            AC_SUBST(MPIFC)
	        ]
	    )
	
	    if test x = x"$MPILIBS"; then
	        AC_LANG_CASE(
	            [C], [
	                AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])
	            ], 
	            [C++], [
	                AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])
	            ], 
	            [Fortran 77], [
	                AC_MSG_CHECKING([for MPI_Init]) 
	                AC_TRY_LINK([],[      call MPI_Init], [MPILIBS=" " AC_MSG_RESULT(yes)], [AC_MSG_RESULT(no)])
	            ],
	            [Fortran], [
	                AC_MSG_CHECKING([for MPI_Init]) 
	                AC_TRY_LINK([],[call MPI_Init], [MPILIBS=" " AC_MSG_RESULT(yes)], [AC_MSG_RESULT(no)])
	            ]
	        )
	    fi
	    
	    if test x = x"$MPILIBS"; then
	        AC_CHECK_LIB(mpi, MPI_Init, [MPILIBS="-lmpi"])
	    fi
	
	    if test x = x"$MPILIBS"; then
	        AC_CHECK_LIB(mpich, MPI_Init, [MPILIBS="-lmpich"])
	    fi
	
	    # We have to use AC_TRY_COMPILE and not AC_CHECK_HEADER because the
	    # latter uses $CPP, not $CC (which may be mpicc).
	    AC_LANG_CASE(
	        [C], [
	            if test x != x"$MPILIBS"; then
	                AC_MSG_CHECKING([for mpi.h])
	                export LAMMPICC="$ax_mpi_save_CC"
	                AC_TRY_COMPILE([#include <mpi.h>], 
	                                [],
	                                [AC_MSG_RESULT(yes)], 
	                                [MPILIBS=""
	                                 AC_MSG_RESULT(no)]
	                )
	                unset LAMMPICC
	            fi
	        ], 
	        [C++], [
	            if test x != x"$MPILIBS"; then
	                AC_MSG_CHECKING([for mpi.h])
	                export LAMMPICXX="$ax_mpi_save_CXX"
	                AC_TRY_COMPILE([#include <mpi.h>],
	                                [],
	                                [AC_MSG_RESULT(yes)], 
	                                [MPILIBS=""
	                                AC_MSG_RESULT(no)]
	                )
	                unset LAMMPICXX 
	            fi
	        ]    
	    )
	
	    AC_LANG_CASE(
	        [C], [
	            CC="$ax_mpi_save_CC"
	        ],
	        [C++], [
	            CXX="$ax_mpi_save_CXX"
	        ],
	        [Fortran 77], [
	            F77="$ax_mpi_save_F77"
	        ],
	        [Fortran], [
	            FC="$ax_mpi_save_FC"
	        ]  
	    )
	
	    AC_SUBST(MPILIBS)
	
	    # Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
	    if test x != x"$MPILIBS"; then
			ax_mpi_ok="yes"
	        AC_DEFINE(HAVE_MPI, 1, [Define if you have the MPI library.])
	        :
	    fi
	fi
	AC_MSG_RESULT()
]) # AX_MPI
