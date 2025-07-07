# Use a Python 3.10 slim image
FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

# Install required tools
RUN apt-get update && apt-get install -y \
    git curl build-essential && \
    apt-get clean

# Install uv
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ENV PATH="/root/.local/bin/:$PATH"

# Clone the repository
# RUN git clone https://github.com/cico-rial/overcooked_ai.git .

# Set working directory
WORKDIR /app

# # Create and sync the environment with uv
# RUN uv venv

# RUN uv sync

# RUN .venv\Scripts\activate

# Set the default command (update this if needed)
CMD ["bash"]
