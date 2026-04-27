FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer → cached on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8080

# uvicorn serves the FastAPI app; async I/O lets one worker handle many concurrent
# in-flight LLM calls.  Increase --workers when CPU (PDF rendering) becomes the
# bottleneck rather than I/O wait.
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1"]
