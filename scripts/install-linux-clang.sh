#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != linux ]; then
	echo "Not a Linux build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	sudo apt-get update 

	sudo apt-get install -q -y gcc-5 g++-5 gfortran-5
	
    if [ -e "/usr/bin/gcc" ]; then
        sudo unlink /usr/bin/gcc;
    fi
    if [ -e "/usr/bin/g++" ]; then
        sudo unlink /usr/bin/g++;
    fi
    if [ -e "/usr/bin/gfortran" ]; then
        sudo unlink /usr/bin/gfortran;
    fi

	sudo ln -s /usr/bin/gcc-5 /usr/bin/gcc
	sudo ln -s /usr/bin/g++-5 /usr/bin/g++
	sudo ln -s /usr/bin/gfortran-5 /usr/bin/gfortran

	sudo apt-get install -q -y clang-3.8 clang++-3.8

	sudo ln -s /usr/bin/clang-3.8 /usr/bin/clang
	sudo ln -s /usr/bin/clang++-3.8 /usr/bin/clang++

	sudo apt-get install -y mpich 
	sudo apt-get install -y libmpich-dev

	# wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
	# tar zxvf mpich-3.2.1.tar.gz
	# cd mpich-3.2.1
	# ./configure CC=clang-3.8 CXX=clang++3.8 FC=gfortran-5 --enable-threads=multiple > out 2>&1 & 
	# config_install_mpich_id=$! 
	# while kill -0 "$config_install_mpich_id" >/dev/null 2>&1; do 
	# 	sleep 300
	# 	tail ./out
	# done 
	# echo "MPICH configuration is finished!"
	# rm -fr ./out
	# make -j 2 > out 2>&1 & 
	# make_install_mpich_id=$! 
	# while kill -0 "$make_install_mpich_id" >/dev/null 2>&1; do 
	# 	sleep 300
	# 	tail ./out
	# done 
	# echo "MPICH build is finished!"
	# rm -fr ./out
	# sudo make install > /dev/null
	# cd ..
fi


