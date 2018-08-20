#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != linux ]; then
	echo "Not a Linux build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	sudo apt-get update 
	
	sudo apt-get install -y gcc-5 g++-5 gfortran-5 clang-3.7

	sudo unlink /usr/bin/gcc && sudo ln -s /usr/bin/gcc-5 /usr/bin/gcc
	sudo unlink /usr/bin/g++ && sudo ln -s /usr/bin/g++-5 /usr/bin/g++
	sudo unlink /usr/bin/gfortran && sudo ln -s /usr/bin/gfortran-5 /usr/bin/gfortran

	sudo apt-get install -y mpich 
	sudo apt-get install -y libmpich-dev
	sudo apt-get install -y libflann-dev
	sudo apt-get install -y libflann1.8

	sudo apt-get update
fi


