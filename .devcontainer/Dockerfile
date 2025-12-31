FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    sudo ca-certificates curl git \
    build-essential pkg-config \
    python3 python3-pip python3-venv python3-dev \
    ngspice \
    && rm -rf /var/lib/apt/lists/*

# 建一个和 VSCode Dev Containers 习惯一致的用户：vscode
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

WORKDIR /workspaces/EEsizer
USER vscode

# 让 pip 默认不输出缓存（可选）
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
