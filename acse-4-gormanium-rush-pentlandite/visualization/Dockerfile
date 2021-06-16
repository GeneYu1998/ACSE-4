FROM ubuntu:latest

RUN apt update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends build-essential \
    libgraphviz-dev

WORKDIR /home/plot

ADD Makefile .
ADD graphviz-example.cpp .

RUN make all


