# TRUEWATCH edge service — Phase 0 scaffolding.
# Target: Hugging Face Spaces, free CPU tier. Not the deployment path for a
# real post, which runs the same image on the board at the post.

FROM python:3.11-slim

# OpenCV needs these even in the headless build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Spaces routes to 7860; EDGE_PORT overrides it elsewhere.
ENV EDGE_HOST=0.0.0.0 \
    EDGE_PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# app.py arrives in Phase 1. Until then this image builds but has nothing to serve.
CMD ["sh", "-c", "uvicorn app:app --host $EDGE_HOST --port $EDGE_PORT"]
