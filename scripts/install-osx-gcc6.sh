#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_OS_NAME}" = osx ]; then
  	export CERTIFICATE_P12=Certificate.p12 ;
  	echo $CERTIFICATE_OSX_P12 | base64 — decode > $CERTIFICATE_P12;

	# Create the keychain with a password
    export KEY_CHAIN=ios-build.keychain

	security create-keychain -p travis $KEY_CHAIN
	# Make the custom keychain default, so xcodebuild will use it for signing
	security default-keychain -s $KEY_CHAIN
	# Unlock the keychain
	security unlock-keychain -p travis $KEY_CHAIN

	# Add certificates to keychain and allow codesign to access them
	security import ./scripts/travis/apple.cer -k ~/Library/Keychains/ios-build.keychain -T /usr/bin/codesign
	security import $CERTIFICATE_P12 -k ~/Library/Keychains/ios-build.keychain -P $KEY_PASSWORD -T /usr/bin/codesign

	security set-key-partition-list -S apple-tool:,apple: -s -k travis ios-build.keychain

    # Set keychain locking timeout to 7200 seconds
    security set-keychain-settings -t 7200 -u $KEY_CHAIN	
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
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
