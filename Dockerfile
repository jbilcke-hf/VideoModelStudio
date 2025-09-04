FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Prevent interactive prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    libavfilter-dev \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Verify FFmpeg installation and libraries
RUN ffmpeg -version && \
    ldconfig && \
    python3 -c "import torchcodec; print('torchcodec imported successfully')" || echo "Warning: torchcodec import test failed, but continuing..."

# actually we found a way to put flash attention inside the requirements.txt
# so we are good, we don't need this anymore:
# RUN pip3 install --no-cache-dir -r requirements_without_flash_attention.txt
# RUN pip3 install wheel setuptools flash-attn --no-build-isolation --no-cache-dir

# Copy application files
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python3", "app.py"]