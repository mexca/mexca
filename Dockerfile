# syntax=docker/dockerfile:1
FROM python:3.9
ARG version=stable
# Disable hdf5 version check error that prevents running mexca
ENV HDF5_DISABLE_VERSION_CHECK=2
RUN apt-get -y update
# Install soundfile and opencv prerequisites
RUN apt-get -y install libsndfile1 && apt-get -y install python3-opencv
RUN git clone -b development https://github.com/mexca/mexca.git
WORKDIR /mexca
# Choose whether to install stable or development
RUN if [ "$version" = "devel" ]; then pip install .; else pip install mexca; fi
# Download pretrained models to prevent download every time the container starts
RUN python docker/download_models.py
EXPOSE 8000
ENTRYPOINT [ "mexca-pipeline" ]
