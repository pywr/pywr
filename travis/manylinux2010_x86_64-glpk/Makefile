NAME	:= pywr/manylinux2010_x86_64-glpk
TAG	:= 4.65
IMG	:= ${NAME}:${TAG}
LATEST	:= ${NAME}:latest

build:
	@docker build -f Dockerfile -t ${IMG} .
	@docker tag ${IMG} ${LATEST}
