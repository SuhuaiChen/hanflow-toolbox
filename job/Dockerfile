FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY job/requirements.job.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# code
COPY job /app/job

# only the two data files you want inside the image
RUN mkdir -p /app/data
COPY data/cedict_ts.u8 /app/data/cedict_ts.u8
COPY data/words.csv /app/data/words.csv

# (if you have shared modules outside job/, copy only those .py files too)
# COPY cedict.py hsk_data.py nlp.py pinyin_convert.py translate.py db.py oss.py scraper.py selector.py sentence_of_day_to_mongo.py tts.py /app/

CMD ["python", "job/hanflow_reading_workflow"]
