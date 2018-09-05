#!/usr/bin/env bash
set -e

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
	export CXX=`which mpic++` 
	export CC=`which mpicc`
	export F77=`which mpifort` 
	export FC=`which mpifort`
fi
