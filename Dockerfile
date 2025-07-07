# Use a Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required tools
RUN apt-get update && apt-get install -y \
    git curl build-essential && \
    apt-get clean

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Clone the repository
RUN git clone https://github.com/cico-rial/overcooked_ai.git .

# Create and sync the environment with uv
RUN uv venv && \
    uv sync

# Set the default command (update this if needed)
CMD ["bash"]
