# Use standard Python 3.11 (avoiding 3.14 build errors)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for image processing (OpenCV, rembg, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the FastAPI application with Gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:8000", "--timeout", "600", "--workers", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app"]