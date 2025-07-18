import os
import shutil

source_dir = 'UTKFace/UTKFace'
dest_dir = 'data/faceAging'
os.makedirs(os.path.join(dest_dir, 'trainA'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 'trainB'), exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.endswith('.jpg'):
        try:
            age = int(filename.split('_')[0])
            src_path = os.path.join(source_dir, filename)
            if 15 <= age <= 30:
                shutil.copy(src_path, os.path.join(dest_dir, 'trainA', filename))
            elif age >= 50:
                shutil.copy(src_path, os.path.join(dest_dir, 'trainB', filename))
        except Exception as e:
            print(f"Skipping file {filename}: {e}")
