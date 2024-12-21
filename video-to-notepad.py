import speech_recognition as sr
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment


def voice_to_text(file_path):
    # 🎯 디렉토리와 파일 경로 설정
    directory = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
    input_path = os.path.join(directory, file_path)
    temporal_wav = os.path.join(directory, "temp.wav")

    # 🎤 음성 인식기 초기화
    recognizer = sr.Recognizer()

    # 🎬 비디오/오디오를 WAV로 변환
    if input_path.endswith(".mp4"):
        video = VideoFileClip(input_path)
        video.audio.write_audiofile(temporal_wav)
        video.close()
    elif input_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(input_path)
        audio.export(temporal_wav, format="wav")

    # 🔊 오디오 파일 처리
    with sr.AudioFile(temporal_wav) as source:
        # 🎵 배경 소음 제거
        recognizer.adjust_for_ambient_noise(source)
        # ⏱ 전체 오디오 길이 확인
        audio_length = source.DURATION
        chunk_duration = 30  # 30초씩 처리
        full_text = []  # 텍스트 조각 모음통

        # 🔄 30초 단위로 반복 처리
        for i in range(0, int(audio_length), chunk_duration):
            # 🎯 현재 청크 녹음
            audio = recognizer.record(
                source, duration=min(chunk_duration, audio_length - i)
            )
            text = recognizer.recognize_google(audio, language="en-US")
            full_text.append(text)

        # 🏁 모든 텍스트 조각을 하나로 합치기
        text = " ".join(full_text)

        # 💾 텍스트 파일로 저장
        output_file = os.path.splitext(file_path)[0] + ".txt"
        output_path = os.path.join(directory, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    # 🧹 임시 파일 삭제
    if os.path.exists(temporal_wav):
        os.remove(temporal_wav)

    return text


# 📂 디렉토리에서 MP4/MP3 파일 찾기
directory = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
files = [f for f in os.listdir(directory) if f.endswith((".mp4", ".mp3"))]

# 🎯 파일 처리
if files:
    file = {"path": files[0]}
    result = voice_to_text(file["path"])
    if result:
        print(f"Completed! {files[0]}")
else:
    print("No files found")
