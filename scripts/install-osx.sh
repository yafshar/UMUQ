#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	brew update;

	sudo rm -fr /usr/local/include/c++

	if [ "${TRAVIS_OSX_IMAGE}" = "xcode8" ]; then
		brew install gcc;
		export GCC_VERSION=`gfortran -dumpversion |cut -d. -f1`  
	else
		brew install gcc@6;
		export GCC_VERSION=`gfortran-6 -dumpversion |cut -d. -f1`  
	fi

	brew reinstall grep --with-default-names;
	brew reinstall gnu-sed --with-default-names;

	sudo ln -s /usr/local/bin/gcc-${GCC_VERSION} /usr/bin/gcc
	sudo ln -s /usr/local/bin/g++-${GCC_VERSION} /usr/bin/g++
	sudo ln -s /usr/local/bin/gfortran-${GCC_VERSION} /usr/bin/gfortran
	sudo ln -s /usr/local/bin/cpp-${GCC_VERSION} /usr/bin/cpp

	brew outdated cmake || brew upgrade cmake ;
	brew outdated autoconf || brew upgrade autoconf ;
	brew outdated automake || brew upgrade automake ;
	
	export HOMEBREW_CC=gcc-${GCC_VERSION}
	export HOMEBREW_CXX=g++-${GCC_VERSION}
	export HOMEBREW_CPP=cpp-${GCC_VERSION}
	export HOMEBREW_LD=gcc-${GCC_VERSION}
	export HOMEBREW_FC=gfortran-${GCC_VERSION}

	brew install --build-from-source mpich;
	
	brew update;
fi


