FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo ca-certificates curl git \
    build-essential pkg-config \
    python3 python3-pip python3-venv python3-dev \
    python-is-python3 \
    ngspice \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user (match VS Code devcontainers convention)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

WORKDIR /workspaces/EEsizer
RUN mkdir -p /workspaces/EEsizer && chown -R ${USERNAME}:${USERNAME} /workspaces

USER ${USERNAME}
ENV HOME=/home/${USERNAME}
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
