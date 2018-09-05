#!/usr/bin/env bash
set -e

if [ "${TRAVIS_OS_NAME}" = "osx" ]; then
	export CXX="mpic++" 
	export CC="mpicc"
	export F77="mpifort" 
	export FC="mpifort"
fi