import speech_recognition as sr
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from docx import Document


# 음성을 텍스트로 변환하는 함수 # Function to convert voice to text #
def voice_to_text(original_filename):
    # 📁 파일 경로 설정
    directory = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
    input_file = os.path.join(directory, original_filename)
    temp_wav = os.path.join(directory, "temp.wav")

    # 🎤 음성 인식기 설정
    recognizer = sr.Recognizer()

    # 🔄 MP4/MP3 파일을 WAV로 변환
    if input_file.endswith(".mp4"):
        video = VideoFileClip(input_file)
        video.audio.write_audiofile(temp_wav)
        video.close()
    elif input_file.endswith(".mp3"):
        audio = AudioSegment.from_mp3(input_file)
        audio.export(temp_wav, format="wav")

    # 🎧 WAV 파일에서 텍스트 추출
    with sr.AudioFile(temp_wav) as source:
        # 🔇 배경 노이즈 제거
        recognizer.adjust_for_ambient_noise(source)

        # ⏱️ 오디오 길이 확인하고 30초 단위로 자르기 위한 설정
        audio_length = source.DURATION
        chunk_duration = 30
        text_pieces = []

        # ✂️ 30초씩 잘라서 텍스트 변환
        for i in range(0, int(audio_length), chunk_duration):
            audio_chunk = recognizer.record(
                source, duration=min(chunk_duration, audio_length - i)
            )
            chunk_text = recognizer.recognize_google(audio_chunk, language="en-US")
            text_pieces.append(chunk_text)

        # 📝 변환된 텍스트 합치기
        full_text = " ".join(text_pieces)

        # 📑 DOCX 파일 생성
        doc = Document()
        doc.add_paragraph(full_text)

        # 💾 DOCX 파일로 저장
        docx_filename = os.path.splitext(original_filename)[0] + ".docx"
        docx_full_location = os.path.join(directory, docx_filename)
        doc.save(docx_full_location)

    # 🗑️ 임시 WAV 파일 삭제
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    return full_text


####Function to convert voice to text Finished####

# 📂 지정된 폴더에서 MP4/MP3 파일 찾기
directory = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
media_files = [f for f in os.listdir(directory) if f.endswith((".mp4", ".mp3"))]

# 🎯 찾은 파일 처리하기
if media_files:
    target_file = {"path": media_files[0]}
    result = voice_to_text(target_file["path"])
    if result:
        print(f"Completed! {media_files[0]}")
else:
    print("No files found")
