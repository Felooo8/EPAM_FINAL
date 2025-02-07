# Use a Python base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY src/ /app/src/
COPY data/ /app/data/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Automatically run the training script when the container starts
CMD ["python3", "src/train/train.py"]
