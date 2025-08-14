import os
import subprocess

directory = r"C:\Users\Jonathan\Documents\kbdocs\text_result"
files = [f for f in os.listdir(directory) if f.endswith(".mp4")]

if not files:
    print("No MP4 files found")
else:
    for file_name in files:
        input_path = os.path.join(directory, file_name)
        output_mp3 = os.path.join(directory, os.path.splitext(file_name)[0] + ".mp3")

        # Skip if MP3 already exists
        if os.path.exists(output_mp3):
            print(f"Skipping {file_name} - MP3 already exists")
            continue

        try:
            print(f"Converting {file_name}...")
            cmd = [
                "ffmpeg",
                "-i",
                input_path,
                "-vn",
                "-acodec",
                "mp3",
                "-ab",
                "128k",
                "-y",
                output_mp3,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Completed: {file_name}")
        except Exception as e:
            print(f"❌ Failed {file_name}: {e}")
