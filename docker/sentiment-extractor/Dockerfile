# syntax=docker/dockerfile-upstream:master-labs
FROM python:3.9-slim
ARG branch=main
# Required to install sentencepiece
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends git
# Required for transformers
ADD --link https://github.com/mexca/mexca.git#${branch} /mexca
WORKDIR /mexca
RUN pip install .[sen]
#Download pretrained models to prevent download every time the container starts
RUN python docker/download_models.py -c sen
ENTRYPOINT ["extract-sentiment"]