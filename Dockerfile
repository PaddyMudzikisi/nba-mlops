FROM python:3.11.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Code is bind-mounted at /app and identified by its git commit; the image only
# carries the environment.
RUN git config --system --add safe.directory /app
ENV PYTHONPATH=/app/src \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1
