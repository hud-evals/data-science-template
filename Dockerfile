# syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS setup

# Update and install core dependencies
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
  vim \
  openssl \
  ca-certificates \
  curl \
  wget \
  sudo \
  bash \
  net-tools \
  novnc \
  x11vnc \
  xvfb \
  python3 \
  python3-pip \
  python3-dev \
  python3-tk \
  python3-wheel \
  python3-venv \
  xfce4 \
  locales \
  libpq5 \
  sqlite3 \
  dbus-x11 \
  xfce4-terminal \
  xfonts-base \
  xdotool \
  psmisc \
  scrot \
  imagemagick \
  pm-utils \
  build-essential \
  python-is-python3 \
  unzip \
  git \
  xauth \
  ffmpeg \
  nginx \
  gnupg \
  gpg \ 
  jq \
  build-essential \
  python3 \
  make \
  gcc \
  g++ \
  libcairo2-dev \
  libjpeg-turbo8-dev \
  libpng-dev \
  libwebp-dev \
  libtiff-dev \
  libgif-dev \
  libvips-dev \
  libgstreamer1.0-0 \
  libgtk-4-1 \
  libgraphene-1.0-0 \
  libwoff1 \
  libevent-2.1-7 \
  libgstreamer-plugins-base1.0-0 \
  libgstreamer-plugins-good1.0-0 \
  libgstreamer-gl1.0-0 \
  libgstreamer-plugins-bad1.0-0 \
  libavif16 \
  libenchant-2-2 \
  libsecret-1-0 \
  libhyphen0 \
  libmanette-0.2-0 \
  libgles2

RUN update-ca-certificates

RUN pip install uv --break-system-packages

WORKDIR /

# Install nvm for ubuntu user
USER ubuntu
ENV HOME=/home/ubuntu

# configure git
RUN git config --global user.email "agent@example.com"
RUN git config --global user.name "mr agent"


# ========================= PROJECT SETUP =========================
# Data Science problem template setup
# =================================================================

# Create the workspace directory
USER root
RUN mkdir -p /home/ubuntu/workspace && chown ubuntu:ubuntu /home/ubuntu/workspace

# Copy all problem templates into the image
COPY --chown=ubuntu:ubuntu ./problem_templates /problem_templates

# Select and set up the workspace with the chosen template
ARG TEMPLATE
RUN if [ -n "$TEMPLATE" ] && [ -d "/problem_templates/$TEMPLATE" ]; then \
      cp -r /problem_templates/$TEMPLATE/* /home/ubuntu/workspace/ && \
      chown -R ubuntu:ubuntu /home/ubuntu/workspace; \
    fi

USER ubuntu
WORKDIR /home/ubuntu/workspace

# Initialize git repo in workspace for agent to work with
RUN git init && git add . && git commit -m "Initial commit" || true

# Set environment variables
ENV HOME=/home/ubuntu \
    DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1.0 \
    DISPLAY_WIDTH=1280 \
    DISPLAY_HEIGHT=800

EXPOSE 6080

# supress AT-SPI errors
ENV NO_AT_BRIDGE=1
USER root

# Setup and start dinit
COPY dinit.d/ /etc/dinit.d/
RUN mkdir -p /var/log/dinit && chmod 755 /var/log/dinit

# Postgres config:
ENV POSTGRES_USER=ubuntu
ENV POSTGRES_PASSWORD=ubuntu
ENV POSTGRES_DB=ubuntu

# ================================ hud evals mcp server setup ================================================
FROM setup AS runtime

# prepare for the hud evals mcp server

# copy python files
COPY ./pyproject.toml /mcp_server/pyproject.toml
COPY ./README.md /mcp_server/README.md
COPY ./tools /mcp_server/tools

ENV RUST_LOG=warn
RUN cd /mcp_server && uv venv && . .venv/bin/activate && uv sync && uv pip install -e .
ENV PYTHONPATH=/mcp_server
ENV PATH=/mcp_server/.venv/bin:$PATH

# env.py and tasks/ live on PYTHONPATH via /mcp_server
COPY ./env.py /mcp_server/env.py
COPY ./tasks /mcp_server/tasks

ENV WIDTH=1280
ENV HEIGHT=800
ENV DISPLAY_NUM=1
RUN mkdir -p /home/ubuntu/screenshots
RUN chmod 777 /home/ubuntu/screenshots
ENV SCREENSHOT_DIR=/home/ubuntu/screenshots
RUN mkdir -p /home/ubuntu/Downloads
RUN chmod 777 /home/ubuntu/Downloads

RUN chmod 777 /root

EXPOSE 6080 3000

ARG HINTS="none"
ENV HINTS=$HINTS

ARG PROBLEM_ID
ENV PROBLEM_ID=$PROBLEM_ID

CMD ["hud", "dev", "env:env", "--stdio"]
