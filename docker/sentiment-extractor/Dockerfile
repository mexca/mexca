# syntax=docker/dockerfile-upstream:master-labs
FROM python:3.9-slim
ARG devel=true
ADD --link https://github.com/mexca/mexca.git#docker-rework /mexca
WORKDIR /mexca
# Choose whether to install stable or development
RUN pip install .[sen] sentencepience
# Download pretrained models to prevent download every time the container starts
RUN python docker/download_models.py -c sen
ENTRYPOINT ["extract-sentiment"]