#
# SYNOPSIS
#
#	AX_MPI([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#	This macro tries to find out how to compile programs that use MPI
#	(Message Passing Interface), a standard API for parallel process
#	communication (see http://www-unix.mcs.anl.gov/mpi/)
#
#	On success, it sets the MPICC, MPICXX, or MPIF77 output variable to
#	the name of the MPI compiler, depending upon the current language.
#	(This may just be $CC/$CXX/$F77, but is more often something like
#	mpicc/mpiCC/mpif77.) It also sets MPILIBS to any libraries that are
#	needed for linking MPI (e.g. -lmpi, if a special
#	MPICC/MPICXX/MPIF77 was not found).
#
#	NOTE: 
#	If you want to compile everything with MPI, you should set:
#
#	CC="$MPICC" #OR# CXX="$MPICXX" #OR# F77="$MPIF77" #OR# FC="$MPIFC"
#	LIBS="$MPILIBS $LIBS"
#
#	The user can force a particular library/compiler by setting the
#	MPICC/MPICXX/MPIF77/MPIFC and/or MPILIBS environment variables.
#	
#	ACTION-IF-FOUND is a list of shell commands to run if an MPI
#	library is found, and ACTION-IF-NOT-FOUND is a list of commands to
#	run it if it is not found. If ACTION-IF-FOUND is not specified, the
#	default action will define HAVE_MPI.
#
#	@category InstalledPackages
#	@author Steven G. Johnson <stevenj@alum.mit.edu>
#	@version 2004-11-05
#	@license GPLWithACException
#
# ADAPTED
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_MPI], [AX_MPI])
AC_DEFUN([AX_MPI], [
	AC_ARG_WITH([mpi], 
		AS_HELP_STRING([--with-mpi@<:@=DIR@:>@], 
			[use MPI library (default is yes) - it is possible to specify the root directory for MPI (optional)]),            
		[
			if test x"$withval" = xno ; then
				AC_MSG_ERROR([ Unable to continue without the MPI library !])
			elif test x"$withval" = xyes ; then
				ac_mpi_path=""
			elif test x"$withval" != x ; then
				ac_mpi_path="$withval"
			else
				ac_mpi_path=
			fi
		], [
			ac_mpi_path=
		]
	)

	dnl if the user does not provide the DIR root directory for MPI, we search the default PATH
	ax_mpi_ok=no

	AS_IF([test x"$ax_mpi_ok" = xno], [  
		AC_MSG_NOTICE(MPI)

		succeeded=no

		ac_mpi_bin=

		if test x"$ac_mpi_path" != x; then
			for ac_mpi_path_tmp in $ac_mpi_path $ac_mpi_path/bin ; do
				for ax_mpi_tmp_CC in mpicc hcc mpcc mpcc_r mpxlc mpxlc_r cmpicc mpigcc tmcc ; do
					if test -x "$ac_mpi_path_tmp"'/'"$ax_mpi_tmp_CC" ; then
						ac_mpi_bin="$ac_mpi_path_tmp"
						break;
					fi
				done
			done
		fi

		AC_PREREQ(2.50)
		
		AS_IF([test x"$ac_mpi_bin" != x], [	
			PATH_SAVED="$PATH"
			PATH="$ac_mpi_bin"':'"$PATH"

			AC_REQUIRE([AC_PROG_CC])
			AC_ARG_VAR(MPICC, [MPI C compiler command])
			MPICC=
			for ac_prog_cc_tmp in mpicc hcc mpcc mpcc_r mpxlc mpxlc_r cmpicc mpigcc tmcc ; do
				AS_VAR_IF(CC, ["$ac_prog_cc_tmp"], [
						MPICC="$CC"
						break;
					], []
				)
				AS_VAR_IF(CC, ["$ac_mpi_bin"'/'"$ac_prog_cc_tmp"], [
						MPICC="$CC"
						break;
					], []
				)
			done
			if test x"$MPICC" = x; then
				AC_CHECK_PROGS(MPICC, mpicc hcc mpcc mpcc_r mpxlc mpxlc_r cmpicc mpigcc tmcc, [no], [$PATH])
			fi
			AS_VAR_IF(MPICC, [no], [AC_MSG_ERROR([Could not find MPI C compiler command !])], 
				[		
					ax_mpi_save_CC="$CC"
					if test -x "$ac_mpi_bin"'/'"$MPICC"; then
						MPICC="$ac_mpi_bin"'/'"$MPICC"
					fi
					CC="$MPICC"         	
					AC_SUBST(MPICC)
				]
			)
			
			AC_REQUIRE([AC_PROG_CXX])
			AC_ARG_VAR(MPICXX, [MPI C++ compiler command])
			MPICXX=
			for ac_prog_cxx_tmp in mpic++ mpicxx mpiCC mpCC hcp mpxlC mpxlC_r cmpic++ cmpic++i mpig++ mpicpc tmCC mpCC_r ; do
				AS_VAR_IF(CXX, ["$ac_prog_cxx_tmp"], [
						MPICXX="$CXX"
						break;
					], []
				)
				AS_VAR_IF(CXX, ["$ac_mpi_bin"'/'"$ac_prog_cxx_tmp"], [
						MPICXX="$CXX"
						break;
					], []
				)
			done
			if test x"$MPICXX" = x; then
				AC_CHECK_PROGS(MPICXX, mpic++ mpicxx mpiCC mpCC hcp mpxlC mpxlC_r cmpic++ cmpic++i mpig++ mpicpc tmCC mpCC_r, [no], [$PATH])
			fi
			AS_VAR_IF(MPICXX, [no], [AC_MSG_ERROR([Could not find MPI C++ compiler command !])], 
				[
					ax_mpi_save_CXX="$CXX"
					if test -x "$ac_mpi_bin"'/'"$MPICXX"; then
						MPICXX="$ac_mpi_bin"'/'"$MPICXX"
					fi  
					CXX="$MPICXX"	
					AC_SUBST(MPICXX)
				]
			)

			AC_LANG_PUSH([C++])
			if test x"$MPILIBS" = x; then
				AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])
			fi

			if test x"$MPILIBS" = x; then
				AC_CHECK_LIB(mpi, MPI_Init, [MPILIBS="-lmpi"])
			fi

			if test x"$MPILIBS" = x; then
				AC_CHECK_LIB(mpich, MPI_Init, [MPILIBS="-lmpich"])
			fi

			# We have to use AC_TRY_COMPILE and not AC_CHECK_HEADER because the
			# latter uses $CPP, not $CC (which may be mpicc).
			if test x"$MPILIBS" != x; then
				AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
								@%:@include <mpi.h>
						]], [[]]
					)], [
						AC_MSG_RESULT(checking mpi.h usability...  yes)
						AC_MSG_RESULT(checking mpi.h presence... yes)
						AC_MSG_RESULT(checking for mpi.h... yes)
						succeeded=yes
					], [
						MPILIBS=
						AC_MSG_RESULT(checking mpi.h usability...  no)
						AC_MSG_RESULT(checking mpi.h presence... no)
						AC_MSG_RESULT(checking for mpi.h... no)
					]
				)
			fi
			AC_LANG_POP([C++])

			PATH="$PATH_SAVED"
		], [	 
			AC_REQUIRE([AC_PROG_CC])
			AC_ARG_VAR(MPICC, [MPI C compiler command])
			MPICC=
			for ac_prog_cc_tmp in mpicc hcc mpcc mpcc_r mpxlc mpxlc_r cmpicc mpigcc tmcc ; do
				AS_VAR_IF(CC, ["$ac_prog_cc_tmp"], [
						MPICC="$CC"
						break;
					], []
				)
			done
			if test x"$MPICC" = x; then
				AC_CHECK_PROGS(MPICC, mpicc hcc mpcc mpcc_r mpxlc mpxlc_r cmpicc mpigcc tmcc, [no])
			fi
			AS_VAR_IF(MPICC, [no], [AC_MSG_ERROR([Could not find MPI C compiler command !])], 
				[
					ax_mpi_save_CC="$CC"
					CC="$MPICC"
					AC_SUBST(MPICC)
				]
			)
			
			AC_REQUIRE([AC_PROG_CXX])
			AC_ARG_VAR(MPICXX, [MPI C++ compiler command])
			MPICXX=
			for ac_prog_cxx_tmp in mpic++ mpicxx mpiCC mpCC hcp mpxlC mpxlC_r cmpic++ cmpic++i mpig++ mpicpc tmCC mpCC_r ; do
				AS_VAR_IF(CXX, ["$ac_prog_cxx_tmp"], [
						MPICXX="$CXX"
						break;
					], []
				)
			done
			if test x"$MPICXX" = x; then
				AC_CHECK_PROGS(MPICXX, mpic++ mpicxx mpiCC mpCC hcp mpxlC mpxlC_r cmpic++ cmpic++i mpig++ mpicpc tmCC mpCC_r, [no])
			fi
			AS_VAR_IF(MPICXX, [no], [AC_MSG_ERROR([Could not find MPI C++ compiler command !])], 
				[
					ax_mpi_save_CXX="$CXX"
					CXX="$MPICXX"
					AC_SUBST(MPICXX)
				]
			)

			AC_LANG_PUSH([C++])
			if test x"$MPILIBS" = x; then
				AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])
			fi

			if test x"$MPILIBS" = x; then
				AC_CHECK_LIB(mpi, MPI_Init, [MPILIBS="-lmpi"])
			fi

			if test x"$MPILIBS" = x; then
				AC_CHECK_LIB(mpich, MPI_Init, [MPILIBS="-lmpich"])
			fi

			# We have to use AC_TRY_COMPILE and not AC_CHECK_HEADER because the
			# latter uses $CPP, not $CC (which may be mpicc).
			if test x"$MPILIBS" != x; then
				AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
						@%:@include <mpi.h>
					]], [[]]
					)], [
						AC_MSG_RESULT(checking mpi.h usability...  yes)
						AC_MSG_RESULT(checking mpi.h presence... yes)
						AC_MSG_RESULT(checking for mpi.h... yes)
						succeeded=yes
					], [
						MPILIBS=
						AC_MSG_RESULT(checking mpi.h usability...  no)
						AC_MSG_RESULT(checking mpi.h presence... no)
						AC_MSG_RESULT(checking for mpi.h... no)
					]
				)
			fi
			AC_LANG_POP([C++])
		])

		CC="$ax_mpi_save_CC"
		CXX="$ax_mpi_save_CXX"

		AC_SUBST(MPILIBS)

		# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
		if test x"$MPILIBS" != x; then
			ax_mpi_ok="yes"
			AC_DEFINE(HAVE_MPI, 1, [Define if you have the MPI library.])
			:
		fi
	])

	AS_IF([test x"$ax_mpi_ok" != xyes], [ AC_MSG_ERROR([ Unable to find the MPI library !])],
		[
			AC_MSG_RESULT()
		]
	)
]) # AX_MPI
