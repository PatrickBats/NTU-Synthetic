import os

folder = '/home/patrick/ssd/discover-hidden-visual-concepts/PatrickProject/fullfolder'

# recursively walk folder and count files
file_count = 0
for dirpath, dirnames, filenames in os.walk(folder):
    file_count += len(filenames)

print(f"Total files in '{folder}': {file_count}")
