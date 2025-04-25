import re
import os
import matplotlib.pyplot as plt

log_file = '/home/user/Desktop/junxie9/skin_cancer_4486/result/logs/DenseNet201_84_20250425_132209.log'
save_dir = '/home/user/Desktop/junxie9/skin_cancer_4486/result'
os.makedirs(save_dir, exist_ok=True)

model_name = os.path.splitext(os.path.basename(log_file))[0]

acc_save_path = os.path.join(save_dir, f'Train_and_Validation_Accuracy_over_Epochs_{model_name}.png')
loss_save_path = os.path.join(save_dir, f'Train_and_Validation_Loss_over_Epochs_{model_name}.png')

epochs = []
train_accs = []
val_accs = []

train_losses = []
val_losses = []

pattern = re.compile(
    r"Epoch\s+(\d+)\s+\| Train Loss: ([\d.]+) \| Train Acc: ([\d.]+) \| Val Loss: ([\d.]+) \| Val Acc: ([\d.]+)"
)

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            train_acc = float(match.group(3))
            val_loss = float(match.group(4))
            val_acc = float(match.group(5))
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accs, label='Train Accuracy')
plt.plot(epochs, val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(acc_save_path)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_save_path)
plt.close()

print(f"Accuracy plot saved to: {acc_save_path}")
print(f"Loss plot saved to: {loss_save_path}")