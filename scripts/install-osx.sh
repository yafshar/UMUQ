#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	if [ "${TRAVIS_OSX_IMAGE}" = "xcode8" ] || [ "${TRAVIS_OSX_IMAGE}" = "xcode8.3" ] ; then
		brew install gcc@6 > out 2>&1 &
 		brew_install_gcc_id=$!
		while kill -0 "$brew_install_gcc_id" > /dev/null 2>&1; do
			sleep 300
			tail ./out
		done
 		echo "GCC installation is finished!"
		rm -fr ./out
		export GCC_VERSION=`gfortran-6 -dumpversion | cut -d. -f1`  
	fi

	brew reinstall grep --with-default-names;
	brew reinstall gnu-sed --with-default-names;
	
	brew outdated cmake || brew upgrade cmake ;
	brew outdated autoconf || brew upgrade autoconf ;
	brew outdated automake || brew upgrade automake ;
	brew outdated wget || brew upgrade wget ;
	
	if [ "${TRAVIS_OSX_IMAGE}" = "xcode8" ] || [ "${TRAVIS_OSX_IMAGE}" = "xcode8.3" ] ; then
		wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
		tar zxvf mpich-3.2.1.tar.gz
		cd mpich-3.2.1
		./configure CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} FC=gfortran-${GCC_VERSION} --enable-threads=multiple > out 2>&1 & 
		config_install_mpich_id=$! 
		while kill -0 "$config_install_mpich_id" > /dev/null 2>&1; do 
			sleep 300
			tail ./out
		done 
		echo "MPICH configuration is finished!"
		rm -fr ./out
		make -j 4 > out 2>&1 & 
		make_install_mpich_id=$! 
		while kill -0 "$make_install_mpich_id" > /dev/null 2>&1; do 
			sleep 300
			tail ./out
		done 
		echo "MPICH build is finished!"
		rm -fr ./out
		sudo make install > /dev/null
		cd ..
	else
		brew install mpich
	fi
	
	brew update;
fi
