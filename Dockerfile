# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Prophet and other libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libpython3-dev \
    libatlas-base-dev \
    libgomp1 \
    gcc \
    g++ \
    curl \
    git \
    wget \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    && apt-get clean

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
