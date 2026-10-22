FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY main.py .
COPY data/ ./data/
CMD ["python", "main.py"]
