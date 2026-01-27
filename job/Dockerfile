FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY job/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# code
COPY job /app/job

# only the two data files you want inside the image
RUN mkdir -p /app/data
COPY data/cedict_ts.u8 /app/data/cedict_ts.u8
COPY data/words.csv /app/data/words.csv

# bake LTP model into the image (local folder)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='LTP/small', local_dir='/app/models/ltp-small', local_dir_use_symlinks=False)"

# (optional but recommended) tell code where the model lives
ENV LTP_MODEL_DIR=/app/models/ltp-small

CMD ["python", "job/hanflow_reading_workflow.py"]
