#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != linux ]; then
	echo "Not a Linux build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	sudo apt-get update 

    if [ -e "/usr/bin/clang" ]; then
        sudo unlink /usr/bin/clang;
    fi
    if [ -e "/usr/bin/clang++" ]; then
        sudo unlink /usr/bin/clang++;
    fi
    if [ -e "/usr/bin/gfortran" ]; then
        sudo unlink /usr/bin/gfortran;
    fi

	sudo ln -s `which clang-5` /usr/bin/clang
	sudo ln -s `which clang++-5` /usr/bin/clang++
	sudo ln -s /usr/bin/gfortran-5 /usr/bin/gfortran

	wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
	tar zxvf mpich-3.2.1.tar.gz
	cd mpich-3.2.1
	./configure CC=clang-5 CXX=clang++-5 FC=gfortran-5 --enable-threads=multiple > out 2>&1 & 
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
fi


