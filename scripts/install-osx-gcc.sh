#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	# # Create the keychain with a password
	# security create-keychain -p travis ios-build.keychain

	# # Make the custom keychain default, so xcodebuild will use it for signing
	# security default-keychain -s ios-build.keychain

	# # Unlock the keychain
	# security unlock-keychain -p travis ios-build.keychain

	# # Add certificates to keychain and allow codesign to access them
	# security import ./Provisioning/certs/apple.cer -k ~/Library/Keychains/ios-build.keychain -T /usr/bin/codesign
	# security import ./Provisioning/certs/distribution.cer -k ~/Library/Keychains/ios-build.keychain -T /usr/bin/codesign
	# security import ./Provisioning/certs/distribution.p12 -k ~/Library/Keychains/ios-build.keychain -P $KEY_PASSWORD -T /usr/bin/codesign

	# security set-key-partition-list -S apple-tool:,apple: -s -k travis ios-build.keychain
	
	brew update;

	sudo rm -fr /usr/local/include/c++

	brew install gcc;

	export GCC_VERSION=`gfortran -dumpversion |cut -d. -f1`
	export CC=`which gcc-${GCC_VERSION}`
	export CXX=`which g++-${GCC_VERSION}`
	export FC=`which gfortran-${GCC_VERSION}`

	# (cd /usr/local && sudo chown -R $(whoami) bin etc include lib sbin share var opt Cellar Caskroom Frameworks)
    
	brew reinstall gnu-sed --with-default-names;
	brew reinstall grep --with-default-names;

	brew outdated cmake || brew upgrade cmake ;
	brew outdated autoconf || brew upgrade autoconf ;
	brew outdated automake || brew upgrade automake ;
	brew outdated wget || brew upgrade wget ;

	brew update
	brew doctor

	# brew reinstall --cc=gcc-${GCC_VERSION} --build-from-source mpich
	# brew install mpich
	
	wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
	tar zxvf mpich-3.2.1.tar.gz
	(cd mpich-3.2.1 && ./configure CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} FC=gfortran-${GCC_VERSION} --enable-threads=multiple > /dev/null && make -j 2 && sudo make install > /dev/null)
	
	brew update;
fi

	# export HOMEBREW_CC=gcc-${GCC_VERSION}
	# export HOMEBREW_CXX=g++-${GCC_VERSION}
	# export HOMEBREW_CPP=cpp-${GCC_VERSION}
	# export HOMEBREW_LD=gcc-${GCC_VERSION}
	# export HOMEBREW_FC=gfortran-${GCC_VERSION}
