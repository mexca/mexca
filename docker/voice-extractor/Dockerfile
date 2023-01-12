# syntax=docker/dockerfile-upstream:master-labs
FROM python:3.9-slim
ARG devel=true
ADD --link https://github.com/mexca/mexca.git#docker-rework /mexca
WORKDIR /mexca
RUN pip install .[voi]
ENTRYPOINT ["extract-voice"]