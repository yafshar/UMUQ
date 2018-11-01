#!/usr/bin/env bash
set -eu

if [ "${TRAVIS_OS_NAME}" != osx ]; then
	echo "Not an OSX build"
	echo "Skipping installation"
	exit 0
fi

if [ "${TRAVIS_SUDO}" = "true" ]; then
	brew update;
	brew update;

    os=$(sw_vers -productVersion | awk -F. '{print $1 "." $2}')
    if softwareupdate --history | grep --silent "Command Line Tools.*${os}"; then
       echo 'Command-line tools already installed.' 
    else
       echo 'Installing Command-line tools...'
       in_progress=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
       sudo touch ${in_progress}
       product=$(softwareupdate --list | awk "/\* Command Line.*${os}/ { sub(/^   \* /, \"\"); print }")
       sudo softwareupdate --verbose --install "${product}" || echo 'Installation failed.' 1>&2 && rm ${in_progress} && exit 1
       sudo rm ${in_progress}
       echo 'Installation succeeded.'
    fi

	sudo rm -fr /usr/local/include/c++

	brew install gcc;
	export GCC_VERSION=`gfortran -dumpversion |cut -d. -f1` 

	export HOMEBREW_CC=gcc-${GCC_VERSION}
	export HOMEBREW_CXX=g++-${GCC_VERSION}
	export HOMEBREW_FC=gfortran-${GCC_VERSION}
	export HOMEBREW_CPP=cpp-${GCC_VERSION}

	(cd /usr/local && sudo chown -R $(whoami) bin etc include lib sbin share var opt Cellar Caskroom Frameworks)

	brew reinstall grep --with-default-names;
	brew reinstall gnu-sed --with-default-names;

	brew outdated cmake || brew upgrade cmake ;
	brew outdated autoconf || brew upgrade autoconf ;
	brew outdated automake || brew upgrade automake ;
	brew outdated wget || brew upgrade wget ;

	brew install mpich --build-from-source
	
	# wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
	# tar zxvf mpich-3.2.1.tar.gz
	# (cd mpich-3.2.1 && ./configure CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} FC=gfortran-${GCC_VERSION} --enable-threads=multiple > /dev/null && make -j 2 && sudo make install > /dev/null)
	
	# brew update;
fi

	# export HOMEBREW_CC=gcc-${GCC_VERSION}
	# export HOMEBREW_CXX=g++-${GCC_VERSION}
	# export HOMEBREW_CPP=cpp-${GCC_VERSION}
	# export HOMEBREW_LD=gcc-${GCC_VERSION}
	# export HOMEBREW_FC=gfortran-${GCC_VERSION}
