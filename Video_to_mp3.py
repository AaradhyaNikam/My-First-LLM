import os
import subprocess

files = os.listdir('Videos')
for file in files :
    tutorial_number = file.split(' [')[0].split(' #')[1]
    file_name = file.split(' ｜ ')[0]
    print(file_name, tutorial_number) 
    subprocess.run(["ffmpeg", "-i", f"Videos/{file}", f"Audios/{tutorial_number} - {file_name}.mp3"])