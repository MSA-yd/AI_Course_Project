import os
from PIL import Image

root_dir = r"C:/USERS/ACER/DESKTOP/CS4486/TOPIC_5_DATA/ISIC84by84"
output_file = r"C:/Users/Acer/Desktop/cs4486/image_sizes.txt"

def check_image_sizes(directory, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    filepath = os.path.join(subdir, file)
                    try:
                        with Image.open(filepath) as img:
                            width, height = img.size
                            line = f"{filepath} - Width: {width}, Height: {height}\n"
                            print(line, end='')  
                            f.write(line)        
                    except Exception as e:
                        error_line = f"Cannot open image {filepath}, error: {e}\n"
                        print(error_line, end='')
                        f.write(error_line)

if __name__ == "__main__":
    check_image_sizes(root_dir, output_file)