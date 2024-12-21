import random
import time
import os
import shutil

directory = r"C:\Users\Jonathan\Documents\kbdocs\auto"


num = 1
f = open(f"diary{num}.txt", "w", encoding="utf-8")
f.write("Tired...")
f.close()


for i in range(1, 101):
    with open(f"{directory}/diary{i}.txt", "w", encoding="utf-8") as f:
        f.write(f"{random.randint(1,101)}patient")

time.sleep(10)

for i in range(1, 101):
    old_file = f"memo{i}.txt"
    new_file = f"diary{i}.txt"
    os.rename(old_file, new_file)
    print(f"{old_file} changed to {new_file}")

for i in range(1, 101):
    os.rename(f"diary{i}.txt", f"memo{i}.txt")
    print(f"old name changed to new name")


# shutil.move("../memo1.txt", "./auto/")
