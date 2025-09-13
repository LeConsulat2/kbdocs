import os
import math
import numpy as np
from pydub import AudioSegment  # (kept because your original code used it)
import whisper
from docx import Document

# from dotenv import load_dotenv   # not needed now
import textwrap

# =========================
# Config
# =========================
USE_CHUNKING = (
    False  # False = simplest path (no splitting). True = in-memory chunking (~10 min)
)
SAMPLE_RATE = 16000  # whisper.load_audio outputs 16k mono float32
CHUNK_SECONDS = 10 * 60
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE

# =========================
# Model
# =========================
print("[INFO] Loading Whisper model (medium.en) on CPU...")
whisper_model = whisper.load_model("medium.en", device="cpu")
print("[INFO] Whisper model loaded successfully!")


# =========================
# Helpers
# =========================
def format_transcription(transcription, line_length=90):
    """
    Formats transcription text for readability:
    - Adds line breaks for easier reading.
    - Combines short sentences with contextually related lines.
    - Keeps keywords like 'Yes', 'Perfect' on the same line as the related sentence.
    """
    keywords_to_combine = [
        "Yes",
        "Yep",
        "Perfect",
        "Okay",
        "Fantastic",
        "All right",
        "Cool",
    ]
    sentences = transcription.replace("\n", " ").split(". ")
    formatted_text = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if any(sentence.startswith(keyword) for keyword in keywords_to_combine):
            formatted_text = formatted_text.rstrip()
            formatted_text += f" {sentence}\n\n"
        else:
            wrapped = textwrap.fill(sentence, width=line_length)
            formatted_text += wrapped + "\n\n"

    return formatted_text.strip()


def save_transcription_to_docx(transcription, output_file):
    print(f"[INFO] Saving transcription to '{output_file}'...")
    doc = Document()
    doc.add_paragraph(transcription)
    doc.save(output_file)
    print(f"[INFO] Transcription saved successfully!")


# =========================
# (ORIGINAL DISK-BASED HELPERS) — NOW UNUSED
# Kept for reference; we no longer call these.
# =========================
# def convert_to_wav(file_path, output_directory):
#     print(f"[INFO] Converting '{file_path}' to WAV format...")
#     audio = AudioSegment.from_file(file_path)
#     wav_file_path = os.path.join(
#         output_directory, os.path.splitext(os.path.basename(file_path))[0] + ".wav"
#     )
#     audio.export(wav_file_path, format="wav")
#     print(f"[INFO] Conversion complete. Saved as '{wav_file_path}'")
#     return wav_file_path
#
# def split_audio(file_path, chunk_length_ms=10 * 60 * 1000):
#     print(
#         f"[INFO] Splitting audio file '{file_path}' into chunks of {chunk_length_ms // 60000} minutes..."
#     )
#     audio = AudioSegment.from_file(file_path)
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunks.append(audio[i : i + chunk_length_ms])
#     print(f"[INFO] Splitting complete. Total chunks: {len(chunks)}")
#     return chunks
#
# def transcribe_audio_chunk(chunk, chunk_index):
#     temp_file = f"temp_chunk_{chunk_index}.wav"
#     print(f"[INFO] Exporting chunk {chunk_index + 1} to temporary WAV file...")
#     chunk.export(temp_file, format="wav")
#     print(f"[INFO] Transcribing chunk {chunk_index + 1}...")
#     result = whisper_model.transcribe(temp_file)
#     os.remove(temp_file)
#     print(f"[INFO] Chunk {chunk_index + 1} transcription complete.")
#     return format_transcription(result["text"])
#
# def process_large_audio_DISK_TEMP(file_path, output_directory):
#     # ORIGINAL PIPELINE (convert to wav -> split to temp wavs -> transcribe each -> clean up)
#     wav_file_path = convert_to_wav(file_path, output_directory)
#     chunks = split_audio(wav_file_path)
#     full_transcript = ""
#     for index, chunk in enumerate(chunks):
#         print(f"[INFO] Processing chunk {index + 1}/{len(chunks)}...")
#         chunk_transcript = transcribe_audio_chunk(chunk, index)
#         full_transcript += f"\n[Chunk {index + 1}]\n{chunk_transcript}"
#     print(f"[INFO] Deleting temporary WAV file: {wav_file_path}...")
#     os.remove(wav_file_path)
#     print(f"[INFO] Temporary WAV file deleted.")
#     return full_transcript


# =========================
# NEW: No-WAV, two modes
# =========================
def process_large_audio_no_split(file_path: str) -> str:
    """
    Simplest path: feed original file (mp3/m4a/mp4) straight to Whisper.
    No conversion, no splitting, no temp files.
    """
    print(f"[INFO] Transcribing full file via Whisper (no WAV/temp files): {file_path}")
    result = whisper_model.transcribe(file_path)  # ffmpeg used under the hood
    return format_transcription(result["text"])


def process_large_audio_chunked_in_memory(file_path: str) -> str:
    """
    Keeps your ~10-minute chunking, but entirely in-memory:
    - Decode with whisper.load_audio (ffmpeg) → 16k mono float32 numpy array
    - Slice by samples, pass numpy chunks directly to transcribe()
    - No temp WAV exports
    """
    print(f"[INFO] Loading audio (ffmpeg) with whisper.load_audio: {file_path}")
    audio = whisper.load_audio(file_path)  # float32 mono, resampled to 16k
    total_samples = audio.shape[-1]
    num_chunks = math.ceil(total_samples / CHUNK_SAMPLES)
    print(
        f"[INFO] Splitting into {num_chunks} chunk(s) of ~{CHUNK_SECONDS//60} minutes"
    )

    full_transcript_parts = []
    for i in range(num_chunks):
        start = i * CHUNK_SAMPLES
        end = min(start + CHUNK_SAMPLES, total_samples)
        chunk = audio[start:end]

        print(
            f"[INFO] Transcribing chunk {i+1}/{num_chunks} "
            f"({start/SAMPLE_RATE:.1f}s – {end/SAMPLE_RATE:.1f}s)"
        )
        result = whisper_model.transcribe(chunk)
        full_transcript_parts.append(
            f"\n[Chunk {i+1}]\n{format_transcription(result['text'])}"
        )

    return "".join(full_transcript_parts)


def process_large_audio(file_path: str, output_directory: str) -> str:
    """
    Entry point used by the main loop.
    Selects between no-split or in-memory chunked flow.
    """
    if USE_CHUNKING:
        return process_large_audio_chunked_in_memory(file_path)
    else:
        return process_large_audio_no_split(file_path)


# =========================
# Main
# =========================
if __name__ == "__main__":
    directory = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
    media_files = [
        f for f in os.listdir(directory) if f.endswith((".mp4", ".mp3", ".m4a"))
    ]

    if not media_files:
        print("[WARNING] No media files found in the directory.")
    else:
        for media_file in media_files:
            audio_file_path = os.path.join(directory, media_file)
            print(f"[INFO] Processing file: {audio_file_path}")
            transcript = process_large_audio(audio_file_path, directory)
            print("[INFO] Transcription completed!")

            output_file_name = os.path.splitext(media_file)[0] + ".docx"
            output_file_path = os.path.join(directory, output_file_name)
            save_transcription_to_docx(transcript, output_file_path)
            print(f"[INFO] Transcription saved to: {output_file_path}")
