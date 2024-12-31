import os
from pydub import AudioSegment
import whisper
from docx import Document
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Whisper 모델 로드 (Medium 모델, CPU 모드)
print("[INFO] Loading Whisper model (medium.en)...")
whisper_model = whisper.load_model("medium.en", device="cpu")
print("[INFO] Whisper model loaded successfully!")


# 1. 오디오 파일을 WAV로 변환
def convert_to_wav(file_path, output_directory):
    print(f"[INFO] Converting '{file_path}' to WAV format...")
    audio = AudioSegment.from_file(file_path)
    wav_file_path = os.path.join(
        output_directory, os.path.splitext(os.path.basename(file_path))[0] + ".wav"
    )
    audio.export(wav_file_path, format="wav")
    print(f"[INFO] Conversion complete. Saved as '{wav_file_path}'")
    return wav_file_path


# 2. 오디오 파일을 10분 단위로 나누는 함수
def split_audio(file_path, chunk_length_ms=10 * 60 * 1000):
    print(
        f"[INFO] Splitting audio file '{file_path}' into chunks of {chunk_length_ms // 60000} minutes..."
    )
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunks.append(audio[i : i + chunk_length_ms])
    print(f"[INFO] Splitting complete. Total chunks: {len(chunks)}")
    return chunks


# 3. Whisper로 오디오 청크 텍스트 변환
def transcribe_audio_chunk(chunk, chunk_index):
    temp_file = f"temp_chunk_{chunk_index}.wav"
    print(f"[INFO] Exporting chunk {chunk_index + 1} to temporary WAV file...")
    chunk.export(temp_file, format="wav")  # WAV 파일로 임시 저장
    print(f"[INFO] Transcribing chunk {chunk_index + 1}...")
    result = whisper_model.transcribe(temp_file)
    os.remove(temp_file)  # 임시 파일 삭제
    print(f"[INFO] Chunk {chunk_index + 1} transcription complete.")
    return result["text"]


# 4. 텍스트를 DOCX 파일로 저장
def save_transcription_to_docx(transcription, output_file):
    print(f"[INFO] Saving transcription to '{output_file}'...")
    doc = Document()
    doc.add_paragraph(transcription)
    doc.save(output_file)
    print(f"[INFO] Transcription saved successfully!")


# 5. 전체 오디오 파일 처리
def process_large_audio(file_path, output_directory):
    # WAV 파일 변환
    wav_file_path = convert_to_wav(file_path, output_directory)

    # WAV 파일을 10분 단위로 나누기
    chunks = split_audio(wav_file_path)

    # 모든 청크를 텍스트로 변환
    full_transcript = ""
    for index, chunk in enumerate(chunks):
        print(f"[INFO] Processing chunk {index + 1}/{len(chunks)}...")
        chunk_transcript = transcribe_audio_chunk(chunk, index)
        full_transcript += f"\n[Chunk {index + 1}]\n{chunk_transcript}"

    # 작업 완료 후 WAV 파일 삭제
    print(f"[INFO] Deleting temporary WAV file: {wav_file_path}...")
    os.remove(wav_file_path)
    print(f"[INFO] Temporary WAV file deleted.")

    return full_transcript


# 메인 실행
directory = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
media_files = [f for f in os.listdir(directory) if f.endswith((".mp4", ".mp3"))]

if not media_files:
    print("[WARNING] No media files found in the directory.")
else:
    for media_file in media_files:
        audio_file_path = os.path.join(directory, media_file)
        print(f"[INFO] Processing file: {audio_file_path}")

        # 1. 오디오 파일 텍스트 변환
        transcript = process_large_audio(audio_file_path, directory)
        print("[INFO] Transcription completed!")

        # 2. DOCX 파일로 저장
        output_file_name = os.path.splitext(media_file)[0] + ".docx"
        output_file_path = os.path.join(directory, output_file_name)
        save_transcription_to_docx(transcript, output_file_path)
        print(f"[INFO] Transcription saved to: {output_file_path}")
