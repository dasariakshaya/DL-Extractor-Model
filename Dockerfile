# Base image
FROM python:3.11-slim

WORKDIR /app

# Copy everything
COPY . .

# Install system dependencies (needed for OpenCV, PaddleOCR, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Start with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "flask_server:app"]
