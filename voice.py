import os
from moviepy.editor import VideoFileClip


def video_to_mp3(file_path):
    directory = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
    input_path = os.path.join(directory, file_path)
    output_mp3 = os.path.join(directory, os.path.splitext(file_path)[0] + ".mp3")

    if input_path.endswith(".mp4"):
        video = VideoFileClip(input_path)
        video.audio.write_audiofile(output_mp3)
        video.close()
        return True


directory = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
files = [f for f in os.listdir(directory) if f.endswith(".mp4")]

if files:
    file = {"path": files[0]}
    result = video_to_mp3(file["path"])
    if result:
        print(f"Completed! {files[0]}")
else:
    print("No files found")
