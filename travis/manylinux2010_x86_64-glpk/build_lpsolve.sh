#!/usr/bin/env bash

LPS_VER="5.5.2.5"
LPS_MD5="3be57261fc41dd8e210f54017220d5f7"

wget https://sourceforge.net/projects/lpsolve/files/lpsolve/${LPS_VER}/lp_solve_${LPS_VER}_source.tar.gz

if [[ "$(md5sum < lp_solve_${LPS_VER}_source.tar.gz)" != "${LPS_MD5}  -" ]]
then
    echo ERROR: Failed to verify lpsolve source.
    exit 1 # terminate and indicate error
fi

SRC_DIR="/app/lp_solve_5.5"
BUILD_DIR=${SRC_DIR}/lpsolve55
# Unpack
tar -xzvf lp_solve_${LPS_VER}_source.tar.gz

patch -p0 < fix-lpsolve-compilation.patch

cd ${BUILD_DIR}
sh ccc

cp ${BUILD_DIR}/bin/ux64/* /usr/lib64/
mkdir /usr/include/lpsolve
cp ${SRC_DIR}/*.h /usr/include/lpsolve/
