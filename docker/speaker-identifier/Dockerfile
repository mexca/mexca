# syntax=docker/dockerfile-upstream:master-labs
FROM python:3.9-slim
ARG HF_TOKEN
ARG branch=main
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends libsndfile1
ADD --link https://github.com/mexca/mexca.git#${branch} /mexca
WORKDIR /mexca
RUN pip install .[spe]
# Download pretrained models to prevent download every time the container starts
RUN python docker/download_models.py -c spe -t $HF_TOKEN
ENTRYPOINT ["identify-speakers"]