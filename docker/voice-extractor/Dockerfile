# syntax=docker/dockerfile-upstream:master-labs
FROM python:3.9-slim
ARG branch=main
ADD --link https://github.com/mexca/mexca.git#${branch} /mexca
WORKDIR /mexca
RUN pip install .[voi]
ENTRYPOINT ["extract-voice"]