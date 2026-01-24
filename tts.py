import os
import dashscope
from dotenv import load_dotenv
import time
import random

load_dotenv()

def _retry(
    fn,
    *,
    max_retries: int = 5,
    base_sleep: float = 0.6,
    max_sleep: float = 8.0,
):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            sleep = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            sleep = sleep * (0.8 + 0.4 * random.random())  # jitter
            time.sleep(sleep)
    raise last_err  # should never reach

# The following URL is for the Singapore region. If you use a model in the China (Beijing) region, replace the URL with: https://dashscope.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

def synthesize_speech(
    *,
    text: str,
    model: str = "qwen3-tts-flash-2025-11-27",
    voice: str = "Kai",
    language_type: str = "Chinese",
    max_retries: int = 5,
    base_sleep: float = 0.6,
) -> str:
    """
    Synthesize speech using DashScope Qwen TTS.
    Returns: temporary WAV/MP3 URL (expires ~24h per DashScope).
    """

    def _call() -> str:
        resp = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
            model=model,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            text=text,
            voice=voice,
            language_type=language_type,
        )

        # defensive parsing (DashScope may return different shapes on error)
        if not isinstance(resp, dict):
            raise RuntimeError(f"Unexpected DashScope response type: {type(resp)}")

        # Some SDKs return {"code": "...", "message": "..."} on error
        if resp.get("code") and resp.get("code") != "200":
            raise RuntimeError(f"DashScope error: code={resp.get('code')} message={resp.get('message')}")

        try:
            url = resp["output"]["audio"]["url"]
        except Exception as e:
            raise RuntimeError(f"Missing audio url in response: {resp}") from e

        if not url:
            raise RuntimeError(f"Empty audio url in response: {resp}")

        return url

    return _retry(_call, max_retries=max_retries, base_sleep=base_sleep)