#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	brew update;
	brew install gcc@5; 
	
	brew outdated cmake || brew upgrade cmake ;
	brew outdated autoconf || brew upgrade autoconf ;
	brew outdated automake || brew upgrade automake ;
	
	brew install --build-from-source --cc=gcc-5 --cxx=g++-5 --fc=gfortran-5 mpich;
	brew install --build-from-source --cc=gcc-5 --cxx=g++-5 flann;
	
	brew update;
fi


