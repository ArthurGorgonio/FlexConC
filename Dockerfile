FROM python:3.10-alpine

ARG USR

LABEL maintainer="Arthur Gorgonio"

RUN echo -e '@edgunity http://nl.alpinelinux.org/alpine/edge/community\n\
  @edge http://nl.alpinelinux.org/alpine/edge/main\n\
  @testing http://nl.alpinelinux.org/alpine/edge/testing\n\
  @community http://dl-cdn.alpinelinux.org/alpine/edge/community'\
  >> /etc/apk/repositories && \
  apk upgrade && \
  apk add --update --no-cache \
  build-base \
  freetype-dev \
  gcc \
  musl-dev \
  openblas-dev && \
  pip install --upgrade pip requests && \
  pip install numpy==1.22

WORKDIR /exp

RUN adduser -D $USR

COPY requirements.txt /exp/

RUN pip install -r requirements.txt

COPY . .
RUN chown -R $USR:$USR /exp

USER $USR

CMD ["python", "main.py"]

