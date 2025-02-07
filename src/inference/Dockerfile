# Use a Python base image
FROM python:3.12

# Define arguments for the model and vectorizer
ARG model_name=best_model.pkl
ARG vectorizer_name=tfidf_vectorizer.pkl

# Set environment variables
ENV MODEL_PATH=/app/models/${model_name}
ENV VECTORIZER_PATH=/app/models/${vectorizer_name}

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY src/ /app/src/
COPY data/ /app/data/
COPY requirements.txt /app/
COPY outputs/models/${model_name} /app/models/${model_name}
COPY outputs/models/${vectorizer_name} /app/models/${vectorizer_name}

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint for the script
ENTRYPOINT ["python3", "src/inference/run_inference.py"]
