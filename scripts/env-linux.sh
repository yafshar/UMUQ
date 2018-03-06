#!/usr/bin/env bash
set -e

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
	if [ "$CXX" = "g++" ]; then 
		export CXX="g++-5" 
		export CC="gcc-5"
		export F77="gfortran-5" 
		export FC="gfortran-5"
	fi
	if [ "$CXX" = "clang++" ]; then 
		export CXX="clang++-3.7" 
		export CC="clang-3.7"
		export F77="gfortran-5" 
		export FC="gfortran-5"
	fi
fi
