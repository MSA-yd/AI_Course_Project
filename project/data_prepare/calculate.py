import os
from PIL import Image
import numpy as np

def compute_mean_std(image_dir):

    img_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_files.append(os.path.join(root, file))

    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    pixel_count = 0

    for idx, img_path in enumerate(img_files):
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img) / 255.0  # 归一化到0~1浮点数

        pixels = img_np.shape[0] * img_np.shape[1]
        pixel_count += pixels

        channel_sum += img_np.sum(axis=(0, 1))
        channel_sum_sq += (img_np ** 2).sum(axis=(0, 1))

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(img_files)} images")

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sum_sq / pixel_count - mean ** 2)

    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    train_dir = "C:/Users/Acer/Desktop/cs4486/Topic_5_Data/ISIC84by84/Train"
    mean, std = compute_mean_std(train_dir)
    print("Mean:", mean)
    print("Std:", std)

    output_path = "C:/Users/Acer/Desktop/cs4486/mean_std.txt"
    with open(output_path, 'w') as f:
        f.write(f"Mean: {mean}\n")
        f.write(f"Std: {std}\n")