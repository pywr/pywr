#!/usr/bin/env bash

GLPK_VER=4.65
wget ftp://ftp.gnu.org/gnu/glpk/glpk-${GLPK_VER}.tar.gz
wget ftp://ftp.gnu.org/gnu/glpk/glpk-${GLPK_VER}.tar.gz.sig

# Verify the source
gpg --recv-keys 5981E818
gpg --verify glpk-4.55.tar.gz.sig glpk-${GLPK_VER}.tar.gz

# Unpack
tar -xzvf glpk-${GLPK_VER}.tar.gz
cd glpk-${GLPK_VER}

# Compile & install
./configure
make
make install
ldconfig --verbose

# Clean up
cd ..
rm -rf glpk-${GLPK_VER}
rm glpk-${GLPK_VER}.tar.gz
rm glpk-${GLPK_VER}.tar.gz.sig
