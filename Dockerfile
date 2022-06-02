
FROM python:3.8-slim

WORKDIR /zinc

ADD zinc.py /zinc
ADD configuration /zinc/configuration

RUN apt-get update && apt-get install -y gcc
RUN  apt-get update && apt-get install -y git gfortran make wget patch

RUN mkdir -m 700 /root/.ssh; \
  touch -m 600 /root/.ssh/known_hosts; \
  ssh-keyscan github.com > /root/.ssh/known_hosts

RUN --mount=type=ssh,id=github git clone git@github.com:utkarshgiri/LensTools.git
RUN  pip install astropy numpy pandas cython
WORKDIR LensTools
RUN python setup.py build && python setup.py install
#RUN cd /zinc
RUN pip install classylss
RUN pip install rich fire
WORKDIR /zinc



