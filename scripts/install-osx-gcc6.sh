#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_OS_NAME}" = osx ]; then
	# Create the keychain with a password
    KEY_CHAIN="ios-build.keychain"
    KEY_CHAIN_PASSWORD="travis"

	security create-keychain -p "$KEY_CHAIN_PASSWORD" "$KEY_CHAIN"
	
	# Make the custom keychain default, so xcodebuild will use it for signing
	security default-keychain -s "$KEY_CHAIN"

	# Unlock the keychain
	security unlock-keychain -p "$KEY_CHAIN_PASSWORD" "$KEY_CHAIN"

	security set-key-partition-list -S apple-tool:,apple: -s -k "$KEY_CHAIN_PASSWORD" ios-build.keychain

    # Set keychain locking timeout to 7200 seconds
    security set-keychain-settings -t 7200 -u "$KEY_CHAIN"
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	brew update 
	
	brew install gcc@6 

	export GCC_VERSION=`gfortran-6 -dumpversion | cut -d. -f1`  

	brew reinstall grep --with-default-names;
	brew reinstall gnu-sed --with-default-names;

	brew outdated cmake || brew upgrade cmake ;
	brew outdated autoconf || brew upgrade autoconf ;
	brew outdated automake || brew upgrade automake ;
	brew outdated wget || brew upgrade wget ;
	
	wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
	tar zxvf mpich-3.2.1.tar.gz
	(cd mpich-3.2.1 && ./configure CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} FC=gfortran-${GCC_VERSION} --enable-threads=multiple > /dev/null && make -j 2 && sudo make install > /dev/null)
	
	brew update;
fi
