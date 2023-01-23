# syntax=docker/dockerfile-upstream:master-labs
FROM python:3.9-slim
ARG devel=true
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends ffmpeg git
ADD --link https://github.com/mexca/mexca.git /mexca
WORKDIR /mexca
RUN pip install .[tra]
# Download pretrained models to prevent download every time the container starts
RUN python docker/download_models.py -c tra
ENTRYPOINT ["transcribe"]