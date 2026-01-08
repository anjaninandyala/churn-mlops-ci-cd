FROM python:3.10-slim

# Set base working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project folders
COPY app /app/app
COPY backend /app/backend
COPY models /app/models
COPY data /app/data
COPY reports /app/reports

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
