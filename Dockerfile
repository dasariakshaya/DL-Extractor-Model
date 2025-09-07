FROM python:3.11-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Pre-download models / initialize to fail early if something's wrong
RUN python -c "import pipeline; recognizer = __import__('pipeline').AdvancedTextRecognizer(); recognizer.initialize_models()"

# Use Gunicorn for production instead of Flask's dev server
# Start your service
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 flask_server:app
