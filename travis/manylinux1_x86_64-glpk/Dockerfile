FROM quay.io/pypa/manylinux1_x86_64

LABEL maintainer="tomo.bbe@gmail.com"

COPY . /app
WORKDIR /app

RUN yum install -y wget
RUN chmod 755 build_lpsolve.sh && ./build_lpsolve.sh
RUN chmod 755 build_glpk.sh && ./build_glpk.sh
