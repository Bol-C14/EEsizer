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


# ---- Langflow (isolated venv) ----
USER root
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install langflow

# make venv default
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# back to non-root user
USER ${USERNAME}

EXPOSE 7860

# Keep data inside container unless you mount /workspaces or /data
ENV LANGFLOW_HOST=0.0.0.0
ENV LANGFLOW_PORT=7860

CMD ["langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
