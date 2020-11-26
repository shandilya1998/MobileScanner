import os
import re

image_dir = 'images'
ann_dir = 'annotations'
raw_ann_dir = 'raw_annotations'

def move_files(folder):
    for f in os.listdir(folder):
        num = re.search(r'[0-9]+', f)
        if num != None:
            num = int(num.group())
            if num <= 800:
                source = os.path.join(folder, f)
                destination = os.path.join(folder, 'train', f)
                os.rename(source, destination)
            else:
                source = os.path.join(folder, f)
                destination = os.path.join(folder, 'val', f)
                os.rename(source, destination)

move_files(image_dir)
move_files(ann_dir)
move_files(raw_ann_dir)
