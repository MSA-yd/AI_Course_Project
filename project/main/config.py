main_dir = '/home/user/Desktop/junxie9/skin_cancer_4486/image/ISIC84by84/Main'
test_dir = '/home/user/Desktop/junxie9/skin_cancer_4486/image/ISIC84by84/Test'
train_dir = '/home/user/Desktop/junxie9/skin_cancer_4486/image/ISIC84by84/train_split'
val_dir = '/home/user/Desktop/junxie9/skin_cancer_4486/image/ISIC84by84/val_split'
result_dir = '/home/user/Desktop/junxie9/skin_cancer_4486/result'

batch_size = 128
num_epochs = 80
learning_rate = 1e-4
model_input_size = (84, 84)

alpha = 1.0
beta = 0.5
gamma = 2
max_m = 0.8
s = 5
add_LDAM_weight = True