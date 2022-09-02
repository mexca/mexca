# syntax=docker/dockerfile:1
FROM python:3.9
WORKDIR /test
ENV HDF5_DISABLE_VERSION_CHECK=2
RUN apt-get -y update
RUN apt-get -y install libsndfile1 && apt-get -y install python3-opencv
RUN pip install mexca
COPY /docker/download_models.py /test/download_models.py
COPY /bin/pipeline.py /test/pipeline.py
COPY /tests/test_files/test_video_audio_5_seconds.mp4 /test/example.mp4
# COPY /examples/docker_test.py /test/docker_test.py
RUN python -m download_models
EXPOSE 8000
ENTRYPOINT [ "python", "-m" , "pipeline" ]
