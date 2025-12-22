FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app/src

WORKDIR /app

# System deps: keep minimal but reliable for geospatial wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libexpat1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY config /app/config

# Create mount points (optional but nice)
RUN mkdir -p /app/input_data /app/derived_data /app/outputs

EXPOSE 8430

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8430", "tbep_invasives.app.wsgi:server"]
