#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != linux ]; then
	echo "Not a Linux build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	sudo apt-get update 

	sudo apt-get install -y gcc-5 g++-5 gfortran-5
	sudo apt-get install -y clang-3.7

	sudo unlink /usr/bin/gfortran && sudo ln -s /usr/bin/gfortran-5 /usr/bin/gfortran

	wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
	tar zxvf mpich-3.2.1.tar.gz
	(cd mpich-3.2.1 && ./configure CC=clang-3.7 CXX=clang++3.7 FC=gfortran-5 --enable-threads=multiple > /dev/null && make -j 2 && sudo make install > /dev/null)

	sudo apt-get update
fi


