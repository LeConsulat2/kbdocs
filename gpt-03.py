"""
pip install openai==1.30.1 whisper pydub ffmpeg-python python-dotenv tiktoken
또는 conda/venv 환경에서 설치 후 사용
"""

import os, re, math, json, time
from datetime import timedelta
from typing import List, Dict

import whisper             # faster-whisper 사용 시 import 변경
from pydub import AudioSegment
import openai
from dotenv import load_dotenv

# ------------------------------------------------------------------ #
# 0. 설정
# ------------------------------------------------------------------ #
MODEL_SIZE          = "medium.en"     # whisper model
MAX_CHARS_PER_LINE  = 42
MAX_SEC_PER_CAPTION = 3.0
BATCH_SIZE          = 8               # 번역 배치 크기
TRANSLATE_MODEL     = "gpt-4o"        # 또는 gpt-4o-mini / gpt-3.5-turbo
LANG_PROMPT = """너는 노마드코더 강의를 번역한다.

규칙:
• 영어 문장은 절대 수정하지 말 것
• 영어는 **굵게** 태그 대신 <b></b> 사용 (SRT 호환)
• 바로 아래 자연스러운 한국어 번역 (해/하는 거야 톤)
• 전문 용어 그대로 (console, deploy 등)
• '사용자' 대신 '유저' 사용
형식:
<b>영어</b>
한국어"""

# ------------------------------------------------------------------ #
# 1. 유틸
# ------------------------------------------------------------------ #
def format_ts(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    hh, rem = divmod(total_ms, 3600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"

def group_segments(segments, max_chars=MAX_CHARS_PER_LINE,
                   max_duration=MAX_SEC_PER_CAPTION) -> List[Dict]:
    """
    Whisper segments → 자막 블록(시작,끝,텍스트)
    """
    caps, cur = [], {"start": segments[0]['start'], "txt": ""}
    for seg in segments:
        candidate = (cur["txt"] + " " + seg["text"]).strip()
        if (len(candidate) > max_chars * 2 or
            seg["end"] - cur["start"] > max_duration):
            # finalize current
            cur["end"] = seg["start"]
            caps.append(cur)
            cur = {"start": seg["start"], "txt": seg['text'].strip()}
        else:
            cur["txt"] = candidate
    cur["end"] = segments[-1]["end"]
    caps.append(cur)
    return caps

def translate_batch(text_list: List[str]) -> List[str]:
    """
    OpenAI 번역 (batch). 실패 시 영어 그대로 반환.
    """
    if not openai.api_key:
        print("[WARN] OPENAI_API_KEY 없음 → 영어만 출력")
        return text_list
    try:
        msgs = [{"role": "system", "content": LANG_PROMPT},
                {"role": "user", "content": "\n\n".join(text_list)}]
        resp = openai.chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=msgs,
            temperature=0.2,
        )
        out = resp.choices[0].message.content
        # 배치 내 각 캡션은 빈 줄로 구분돼있다는 가정
        ko_lines = [blk.strip() for blk in out.split("\n\n") if blk.strip()]
        if len(ko_lines) != len(text_list)*2:  # EN/KO 쌍이 안 맞으면 fallback
            raise ValueError("batch size mismatch")
        # EN/KO 순서 → 뒤쪽만 추출
        return [ko_lines[i*2+1] for i in range(len(text_list))]
    except Exception as e:
        print("[ERROR] 번역 실패:", e)
        return text_list

def save_srt(captions: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for idx, cap in enumerate(captions, 1):
            f.write(str(idx)+"\n")
            f.write(f"{format_ts(cap['start'])} --> {format_ts(cap['end'])}\n")
            f.write(f"<b>{cap['txt_en']}</b>\n{cap['txt_ko']}\n\n")
    print(f"[DONE] SRT saved → {path}")

# ------------------------------------------------------------------ #
# 2. 메인 파이프라인
# ------------------------------------------------------------------ #
def process_video(file_path: str):
    base   = os.path.splitext(file_path)[0]
    wav_fp = base + ".wav"

    # (a) 영상/음성 → wav
    print("[1] Converting to WAV...")
    audio = AudioSegment.from_file(file_path)
    audio.export(wav_fp, format="wav")

    # (b) Whisper 전사
    print("[2] Whisper transcribing...")
    model  = whisper.load_model(MODEL_SIZE)
    result = model.transcribe(wav_fp, word_timestamps=False, verbose=False)
    segments = result["segments"]
    print(f"    Segments: {len(segments)}")

    # (c) 자막 그룹핑
    caps = group_segments(segments)
    print(f"    Captions grouped: {len(caps)}")

    # (d) 번역(batch)
    texts = [c["txt"].strip() for c in caps]
    translated = []
    for i in range(0, len(texts), BATCH_SIZE):
        translated.extend(translate_batch(texts[i:i+BATCH_SIZE]))
        time.sleep(0.5)  # rate-limit 완충

    # (e) 캡션 병합
    for c, ko in zip(caps, translated):
        c["txt_en"] = c["txt"]
        c["txt_ko"] = ko

    # (f) SRT 저장
    srt_path = base + "_bilingual.srt"
    save_srt(caps, srt_path)

    # (g) 청소
    os.remove(wav_fp)

# ------------------------------------------------------------------ #
# 3. 실행
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    load_dotenv()                         # .env 에 OPENAI_API_KEY 저장
    openai.api_key = os.getenv("OPENAI_API_KEY")

    folder = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
    for file in os.listdir(folder):
        if file.lower().endswith((".mp4", ".mp3", ".wav", ".m4a")):
            print("\n=== Processing:", file, "===")
            process_video(os.path.join(folder, file))
