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
	sudo apt-get install -y mpich 
	sudo apt-get install -y libmpich-dev
	sudo apt-get install -y libblas-dev
	sudo apt-get install -y liblapack-dev
	sudo apt-get install -y libgtest-dev 
	sudo apt-get install -y libeigen3-dev
	sudo apt-get install -y libflann-dev
	sudo apt-get install -y libflann1.8
fi


