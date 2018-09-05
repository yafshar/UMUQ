#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	brew update;
	
	brew install gcc;
	GCC_VERSION = `gfortran -dumpversion |cut -d. -f1`  
	
	brew outdated cmake || brew upgrade cmake ;
	brew outdated autoconf || brew upgrade autoconf ;
	brew outdated automake || brew upgrade automake ;
	
	brew install --build-from-source --cc=gcc-${GCC_VERSION} --cxx=g++-${GCC_VERSION} --fc=gfortran-${GCC_VERSION} mpich;
	brew install --build-from-source --cc=gcc-${GCC_VERSION} --cxx=g++-${GCC_VERSION} flann;
	
	brew update;
fi


