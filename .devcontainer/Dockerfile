FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    sudo ca-certificates curl git \
    build-essential pkg-config \
    python3 python3-pip python3-venv python3-dev \
    # Circuit simulator
    ngspice \
    && rm -rf /var/lib/apt/lists/*

# 使用 NodeSource 安装较新的 Node.js (20.x)
# 需要 curl 和 ca-certificates 已经安装好
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 建一个和 VSCode Dev Containers 习惯一致的用户：vscode
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# workspace
WORKDIR /workspaces/EEsizer
RUN mkdir -p /workspaces/EEsizer && chown -R ${USERNAME}:${USERNAME} /workspaces

# ---- Install Python deps as root (global) ----
USER root
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        openai \
        python-dotenv \
        numpy \
        pandas \
        scipy \
        matplotlib \
        pytest

# ---- Install codex CLI (Node) ----
RUN npm install -g @openai/codex

# ---- Switch to vscode for interactive use ----
USER vscode

# PyCharm introspection 很依赖 HOME 是“环境变量”，不要只靠 /etc/passwd
ENV HOME=/home/vscode
ENV PATH="/home/vscode/.local/bin:${PATH}"
