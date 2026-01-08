# -------------------------------
# Dockerfile for Training Pipeline
# -------------------------------

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY ../requirements.txt requirements.txt
COPY ../src src
COPY ../data data
COPY ../models models

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt



# Copy entrypoint script
COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# Run the training pipeline
ENTRYPOINT ["./entrypoint.sh"]
