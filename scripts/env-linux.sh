#!/usr/bin/env bash
set -e

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
	if [ "$CXX" = "g++" ]; then 
		export CXX="mpic++" 
		export CC="mpicc"
		export F77="mpifort" 
		export FC="mpifort"
	fi
	if [ "$CXX" = "clang++" ]; then 
		export CXX="mpic++" 
		export CC="mpicc"
		export F77="mpifort" 
		export FC="mpifort"
	fi
fi
