import matplotlib.pyplot as plt
import numpy as np

classes = ['SCC', 'MEL', 'BKL', 'VASC', 'BCC', 'AK', 'NV', 'DF']

test_counts = [100, 100, 100, 100, 100, 100, 100, 100]
val_counts = [106, 885, 505, 31, 645, 147, 2555, 28]
main_counts = [528, 4422, 2524, 153, 3223, 735, 12775, 139]
train_counts = [422, 3537, 2019, 122, 2578, 588, 10220, 111]

x = np.arange(len(classes))
width = 0.2  

fig, ax = plt.subplots(figsize=(12, 6))

rects1 = ax.bar(x - 1.5*width, test_counts, width, label='Test')
rects2 = ax.bar(x - 0.5*width, val_counts, width, label='Validation')
rects3 = ax.bar(x + 0.5*width, train_counts, width, label='Train')
rects4 = ax.bar(x + 1.5*width, main_counts, width, label='Main')

ax.set_ylabel('Number of Images')
ax.set_xlabel('Classes')
ax.set_title('Image Counts per Class in Different Dataset Splits')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for rects in [rects1, rects2, rects3, rects4]:
    autolabel(rects)

plt.tight_layout()
save_path = '/home/user/Desktop/junxie9/skin_cancer_4486/result/image_counts.png'
plt.savefig(save_path)
plt.show()

print(f"Image saved to: {save_path}")