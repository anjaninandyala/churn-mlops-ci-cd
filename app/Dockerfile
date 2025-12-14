FROM python:3.10-slim

# 1️⃣ Set base working directory
WORKDIR /app

# 2️⃣ Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3️⃣ Copy project folders
COPY app /app/app
COPY models /app/models
COPY data /app/data
COPY reports /app/reports


# 4️⃣ Expose Streamlit port
EXPOSE 8501

# 5️⃣ Run Streamlit with correct path
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
