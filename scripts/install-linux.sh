#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != linux ]; then
	echo "Not a Linux build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	sudo apt-get update 
	
	#sudo update-alternatives --remove-all gcc 
	#sudo update-alternatives --remove-all g++
	#sudo update-alternatives --remove-all gfortran
    
	sudo apt-get install -y gcc-5 g++-5 gfortran-5 clang-3.7

	#sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10
	#sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10
	#sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-5 10

	#sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 20
	#sudo update-alternatives --set cc /usr/bin/gcc

	#sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 20
	#sudo update-alternatives --set c++ /usr/bin/g++

	#sudo update-alternatives --config gcc
	#sudo update-alternatives --config g++
	#sudo update-alternatives --config gfortran

	sudo apt-get install -y mpich 
	sudo apt-get install -y libmpich-dev
	sudo apt-get install -y libblas-dev
	sudo apt-get install -y liblapack-dev
	sudo apt-get install -y liblapacke
	sudo apt-get install -y liblapacke-dev
	sudo apt-get install -y libgtest-dev 

	# This is valid on ubuntu Linux    
	(cd /usr/src/gtest && sudo cmake CMakeLists.txt && sudo make && sudo cp *.a /usr/lib)

	sudo apt-get install -y libeigen3-dev
	sudo apt-get install -y libflann-dev
	sudo apt-get install -y libflann1.8

	# TORC installation 
	cd external/torc_lite
	# Create the configuration script
	autoreconf -i
	# Configure and install
	./configure
	make
	sudo make install
	cd ../../

	sudo apt-get update
fi


