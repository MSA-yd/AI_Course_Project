import os
import shutil
from sklearn.model_selection import train_test_split

main_dir = "/home/user/Desktop/junxie9/skin_cancer_4486/image/ISIC84by84/Main"
train_dir = os.path.join(main_dir)

output_dir = "/home/user/Desktop/junxie9/skin_cancer_4486/image/ISIC84by84"
train_split_dir = os.path.join(output_dir, "train_split")
val_split_dir = os.path.join(output_dir, "val_split")

os.makedirs(train_split_dir, exist_ok=True)
os.makedirs(val_split_dir, exist_ok=True)

categories = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

val_ratio = 0.2  

for category in categories:
    category_path = os.path.join(train_dir, category)
    images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

    train_images, val_images = train_test_split(images, test_size=val_ratio, random_state=42)

    os.makedirs(os.path.join(train_split_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_split_dir, category), exist_ok=True)

    for img in train_images:
        shutil.copy2(os.path.join(category_path, img), os.path.join(train_split_dir, category, img))

    for img in val_images:
        shutil.copy2(os.path.join(category_path, img), os.path.join(val_split_dir, category, img))

print("Dataset split completed!")
print(f"Training data saved to: {train_split_dir}")
print(f"Validation data saved to: {val_split_dir}")