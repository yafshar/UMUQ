#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	brew update;
  
	brew reinstall gnu-sed --with-default-names;
	brew reinstall grep --with-default-names;

	brew outdated cmake || brew upgrade cmake ;
	brew outdated autoconf || brew upgrade autoconf ;
	brew outdated automake || brew upgrade automake ;
	brew outdated wget || brew upgrade wget ;

	brew install mpich
	
	brew update;
fi

	# export HOMEBREW_CC=gcc-${GCC_VERSION}
	# export HOMEBREW_CXX=g++-${GCC_VERSION}
	# export HOMEBREW_CPP=cpp-${GCC_VERSION}
	# export HOMEBREW_LD=gcc-${GCC_VERSION}
	# export HOMEBREW_FC=gfortran-${GCC_VERSION}
