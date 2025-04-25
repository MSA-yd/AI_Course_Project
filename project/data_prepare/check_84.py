import os
from PIL import Image

root_dir = r"C:/USERS/ACER/DESKTOP/CS4486/TOPIC_5_DATA/ISIC84by84"

def check_images_84x84(directory):
    all_correct = True
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                filepath = os.path.join(subdir, file)
                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                        if width != 84 or height != 84:
                            print(f"Image size mismatch: {filepath} - Width: {width}, Height: {height}")
                            all_correct = False
                except Exception as e:
                    print(f"Cannot open image {filepath}, error: {e}")
                    all_correct = False
    if all_correct:
        print("All images are 84x84 pixels.")

if __name__ == "__main__":
    check_images_84x84(root_dir)