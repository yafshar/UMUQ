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
	
	brew outdated cmake || brew upgrade cmake ;
	brew outdated autoconf || brew upgrade autoconf ;
	brew outdated automake || brew upgrade automake ;
	
	export HOMEBREW_CC=gcc-${GCC_VERSION}
	export HOMEBREW_CXX=g++-${GCC_VERSION}
	export HOMEBREW_CPP=cpp-${GCC_VERSION}
	export HOMEBREW_LD=gcc-${GCC_VERSION}
	export HOMEBREW_FC=gfortran-${GCC_VERSION}

	brew install mpich;
	brew install flann;
	
	brew update;
fi


