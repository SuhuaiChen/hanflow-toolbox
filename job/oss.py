import os
import io
import hashlib
import httpx
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from pydub import AudioSegment
from tts import synthesize_speech
from dotenv import load_dotenv

try:
    load_dotenv()
except Exception:
    pass

def build_public_url(key: str) -> str:
    base = os.getenv("R2_PUBLIC_BASE_URL")
    return f"{base.rstrip('/')}/{key}"

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def r2_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def download_bytes(url: str, timeout: float = 30.0) -> bytes:
    # download quickly because DashScope URL expires
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content


def wav_bytes_to_mp3_bytes(wav_bytes: bytes, bitrate: str = "64k") -> bytes:
    """
    Converts WAV bytes to MP3 bytes using ffmpeg via pydub.
    bitrate: "64k" is usually enough for Mandarin speech; use "96k" if you prefer.
    """
    audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")

    out = io.BytesIO()
    audio.export(out, format="mp3", bitrate=bitrate)
    return out.getvalue()


def upload_mp3_to_r2(
    *,
    mp3_bytes: bytes,
    s3,
    key: str,
    content_type: str = "audio/mpeg",
    cache_control: str = "public, max-age=31536000, immutable",
) -> str:
    """
    Uploads mp3 to R2. Returns URL if R2_PUBLIC_BASE_URL is set; otherwise returns the object key.
    """
    bucket = os.environ["R2_BUCKET"]

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=mp3_bytes,
        ContentType=content_type,
        CacheControl=cache_control,
    )

    base = os.getenv("R2_PUBLIC_BASE_URL")
    if base:
        return f"{base.rstrip('/')}/{key}"
    return key

def r2_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def attach_tts_audio(
    *,
    s3,
    text: str,
    voice: str = "Ethan",
    language: str = "zh",
    kind: str = "sentence", 
    bitrate: str = "64k",
    date : str = "",

) -> tuple[str, str]:
    cache_id = sha1_hex(f"{kind}|{voice}|{language}|{text}")

    if date :
        # Use date to organize TTS files by date
        object_key = f"tts/{language}/{voice}/{kind}/{date}/{cache_id}.mp3"
    else:
        object_key = f"tts/{language}/{voice}/{kind}/{cache_id}.mp3"

    bucket = os.environ["R2_BUCKET"]

    if r2_exists(s3, bucket, object_key):
        return build_public_url(object_key)

    wav_url = synthesize_speech(text=text, voice=voice, language_type=language)
    wav_bytes = download_bytes(wav_url)
    mp3_bytes = wav_bytes_to_mp3_bytes(wav_bytes, bitrate=bitrate)
    url = upload_mp3_to_r2(mp3_bytes=mp3_bytes, key=object_key, s3=s3) 

    return url


def delete_prefix(s3, bucket: str, prefix: str, dry_run: bool = False) -> int:
    paginator = s3.get_paginator("list_objects_v2")
    deleted = 0
    batch = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if dry_run:
                print(key)
                continue

            batch.append({"Key": key})
            if len(batch) == 1000:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                deleted += len(batch)
                batch = []

    if (not dry_run) and batch:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)

    return deleted


if __name__ == "__main__":
    text = "你好，欢迎使用汉流工具箱！"
    result = attach_tts_audio(
        s3=r2_client(),
        text=text,
        voice="Ethan",
        language="Chinese",
        kind="sentence",
        bitrate="64k",
    )
    print(f"Public URL {result['url']}")

    # s3 = r2_client()
    # bucket = os.environ["R2_BUCKET"]
    # prefix = "tts/zh/Kai/sentence"
    # deleted_count = delete_prefix(s3, bucket, prefix, dry_run=False)
    # print(f"Deleted {deleted_count} objects with prefix '{prefix}'")