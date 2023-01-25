# syntax=docker/dockerfile-upstream:master-labs
FROM python:3.9-slim
ARG branch=main
ADD --link https://github.com/mexca/mexca.git#${branch} /mexca
WORKDIR /mexca
# Choose whether to install stable or development
RUN pip install .[vid]
# Download pretrained models to prevent download every time the container starts
RUN python docker/download_models.py -c vid
ENTRYPOINT ["extract-faces"]