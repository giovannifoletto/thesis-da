FROM ubuntu:latest

WORKDIR /work

COPY . /work

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip

RUN pip install -r src/requirements.py
