import os

def count_images_per_folder(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    folder_image_counts = {}

    for root, dirs, files in os.walk(folder_path):
        count = 0
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                count += 1
        folder_image_counts[root] = count

    return folder_image_counts

folder = "/home/user/Desktop/junxie9/skin_cancer_4486/image/ISIC84by84"

output_dir = "/home/user/Desktop/junxie9/skin_cancer_4486/result"
os.makedirs(output_dir, exist_ok=True)  

output_file = os.path.join(output_dir, "image_counts.txt")

counts = count_images_per_folder(folder)

with open(output_file, "w", encoding="utf-8") as f:
    for folder_path, num_images in counts.items():
        f.write(f"{folder_path} contains {num_images} images\n")

print(f"Image counts have been saved to {output_file}")