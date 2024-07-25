FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel as base

# Install python, ffmpeg, and poetry
RUN apt-get update && apt-get install -y python3.11 curl ffmpeg
RUN pip install poetry

WORKDIR /workspace

COPY audio_example.mp3 .
COPY Makefile .

# Install dependencies
COPY pyproject.toml .
COPY poetry.lock .

RUN poetry install --no-root

USER root