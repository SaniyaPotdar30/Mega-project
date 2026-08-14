FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    gcc \
    g++ \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch and Torchvision first (much smaller download, prevents network timeout)
RUN pip install --no-cache-dir --retries 5 --timeout 120 torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

# Copy application files
COPY . .

# Expose Flask port
EXPOSE 5000

# Start application
CMD ["python", "app.py"]