# Base image — official Python image, slim version (smaller size)
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (before copying all code)
# This is a Docker best practice — if requirements.txt
# doesn't change, Docker reuses the cached layer
# and skips reinstalling packages on every build
COPY requirements.txt .

# Install dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your application code
COPY . .

# Tell Docker this container listens on port 8000
EXPOSE 8090

# Command that runs when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]