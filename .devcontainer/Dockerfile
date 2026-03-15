FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# Remove problematic Yarn repository if it exists
RUN rm -f /etc/apt/sources.list.d/yarn.list

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Switch to vscode user and install uv and Claude Code
USER vscode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# RUN npm install -g @anthropic-ai/claude-code

# Set working directory
WORKDIR /workspaces/vtt2minutes

# Set default shell
SHELL ["/bin/bash", "-c"]
