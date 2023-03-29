FROM mcr.microsoft.com/devcontainers/python:0-3.7
COPY /storage-01/ml-dritter/PromptToPrompt/test.txt /workspaces/PromptToPrompt/test.txt

RUN ./setup.sh

LABEL version=”0.1”
LABEL description=”First_try”
# USER root
# VOLUME /storage-01/datasets/:/workspaces/PromptToPrompt/datasets
# CMD bashdoc