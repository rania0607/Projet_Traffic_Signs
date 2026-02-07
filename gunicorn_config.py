import multiprocessing
import os

# Bind to Render's PORT or default 10000
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# CRITICAL: Use only 1 worker to avoid loading model multiple times
workers = 1

# Use threads instead of workers for concurrency
threads = 2

# Worker class
worker_class = 'gthread'

# Extended timeout for model loading (120 seconds)
timeout = 120

# Graceful timeout
graceful_timeout = 30

# Keep alive
keepalive = 5

# Preload app (load model once before handling requests)
preload_app = True

# Max requests before worker restart (memory management)
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Memory optimization
worker_tmp_dir = '/dev/shm'  # Use shared memory for temp files