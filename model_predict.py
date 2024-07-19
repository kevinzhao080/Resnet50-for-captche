import torch
from torchvision import transforms
from buildCaptchaModel import CaptchaModel
import PIL
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
from CaptchaDataset import CaptchaDataset
import glob
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt

def single_predict(model, image, decoding_dict, device="cpu"):

    img = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            # transforms.RandomResizedCrop((60, 160), scale=(1.0, 1.0)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.7570), (0.3110))
        ]
    )(image.resize((160, 60)).convert("RGB"))

    if img.dim() == 3:
        img = img.unsqueeze(0)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        out = model(img.to(device))
        probabilities = torch.softmax(out.view(-1, 34), dim=1).view(-1, 4, 34)

    label = []
    encoded_vector = out.reshape(34, 4).argmax(0)
    for key in encoded_vector.detach().cpu().numpy():
        label.append(decoding_dict[key])
    return "".join(label),out,probabilities

batch_size = 128
test_dir = "./images/test/"
test_dataset = CaptchaDataset(test_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
model = CaptchaModel()
model.load_state_dict(torch.load("./weights/ResNet_mytrain_weigthts.pth"))
# ckpt_path = "./lightning_logs/version_3/checkpoints/epoch=34-step=13685.ckpt"
# model = CaptchaModel.load_from_checkpoint(ckpt_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# weights = torch.load('./weights/ResNet_mytrain.pth',map_location='cpu')
# model.load_state_dict(weights)
# C:\Users\zhao\Desktop\captcheregonizer\ResNet_for_captcha\lightning_logs\version_44\checkpoints\epoch=1-step=782.ckpt
# img = PIL.Image.open(os.path.join('./images1/test/', "100000_ZEGE.png"))

# Initialize storage for true labels and predicted probabilities
y_true = []
y_scores = []

dic = {}
right = 0
images = glob.glob('./test/*')
images = [i.split('\\')[1] for i in images]
for image in tqdm(images):
    img = PIL.Image.open(os.path.join('./test/', image))
    answer = image.split('.')[0].split('_')[1]
    dic[answer],out,probabilities = single_predict(model, img, test_dataset.decoding_dict)

    # Convert the ground truth answer to one-hot encoding
    y_true_one_hot = np.zeros((4, 34))
    for idx, char in enumerate(answer):
        char_idx = test_dataset.encoding_dict[char]
        y_true_one_hot[idx, char_idx] = 1

    y_true.append(y_true_one_hot)
    y_scores.append(probabilities.cpu().detach().numpy())

# Convert lists to numpy arrays and flatten them
y_true = np.array(y_true).reshape(-1, 34)
y_scores = np.array(y_scores).reshape(-1, 34)

precision = dict()
recall = dict()
average_precision = dict()
for i in range(4):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
    average_precision[i] = average_precision_score(y_true[:, i], y_scores[:, i])

print(precision)
print(average_precision)

# Plot the precision-recall curve for each class
plt.figure(figsize=(8, 6))
for i in range(4):
    plt.plot(recall[i], precision[i], marker='.', label=f'Class {i} (AP = {average_precision[i]:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# # 计算PR曲线
# precision = dict()
# recall = dict()
# average_precision = dict()

# for i in range(4):  # 针对4个字符位置
#     # 提取当前字符位置的y_true和y_scores
#     current_y_true = y_true[:, i*34:(i+1)*34].flatten()
#     current_y_scores = y_scores[:, i*34:(i+1)*34].flatten()
    
#     # 计算PR曲线和平均精度
#     precision[i], recall[i], _ = precision_recall_curve(current_y_true, current_y_scores)
#     average_precision[i] = average_precision_score(current_y_true, current_y_scores)

# # 绘制PR曲线
# plt.figure(figsize=(8, 6))
# for i in range(4):
#     plt.plot(recall[i], precision[i], marker='.', label=f'Class {i} (AP = {average_precision[i]:0.2f})')

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc="best")
# plt.show()



# # Plot the precision-recall curve
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, marker='.', label='Average precision = {0:0.2f}'.format(average_precision))
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='best')
# plt.show()

for k,v in dic.items():
    print(f"Answer:{k} predict:{v}")
    if(k == v):
        print("right")
        right += 1
    else:
        print("error")
print('{}张图片 rate: {:.2%}'.format(len(images),right/len(images)))

# print(images)
# img.show()

# print(single_predict(model, img, test_dataset.decoding_dict))
