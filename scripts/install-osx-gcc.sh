#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
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
