FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer → cached on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8080

# 2 workers; 120s timeout — PDF rendering for large papers can take a while
CMD ["gunicorn", "server:app", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "2", \
     "--timeout", "120"]
