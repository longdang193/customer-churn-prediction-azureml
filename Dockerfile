# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages
# e.g., RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy the requirements files into the container at /app
COPY requirements.txt dev-requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r dev-requirements.txt

# Copy the rest of the application's source code from your host to your image filesystem.
COPY . /app/

# Inform Docker that the container listens on the specified port at runtime.
# EXPOSE 8000

# Define the command to run your app
# This will be overridden by AML, but is good practice for local testing
# CMD ["python", "src/train.py"]


