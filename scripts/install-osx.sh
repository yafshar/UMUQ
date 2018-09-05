#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	brew update;

	brew uninstall gnu-sed;
	brew install gnu-sed --with-default-names;

	brew install gcc;
	export GCC_VERSION=`gfortran -dumpversion |cut -d. -f1`  
	
	sudo unlink /usr/bin/gcc && sudo ln -s /usr/local/bin/gcc-${GCC_VERSION} /usr/bin/gcc
	sudo unlink /usr/bin/g++ && sudo ln -s /usr/local/bin/g++-${GCC_VERSION} /usr/bin/g++
	sudo unlink /usr/bin/gfortran && sudo ln -s /usr/local/bin/gfortran-${GCC_VERSION} /usr/bin/gfortran

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


