import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from sklearn.model_selection import train_test_split

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transforms)
test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda")

X_train = []
y_train = []
for images, labels in train_loader:
    X_train.append(images.view(images.size(0), -1))
    y_train.append(labels)
X_train = torch.cat(X_train, dim=0).to(device)
y_train = torch.cat(y_train, dim=0).to(device)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# SVM
def svm_loss(X, y, W, reg):
    N = X.shape[0]
    scores = X.mm(W) # 形状为（N, C）,N:样本数, C:类别数
    # 每个样本在其正确类别上的得分
    correct_class_scores = scores[torch.arange(N), y].view(-1, 1)   # 形状为（N, 1)
    # 计算损失
    margins = torch.max(torch.zeros_like(scores), scores - correct_class_scores + 1)
    margins[torch.arange(N), y] = 0 # 不计算正确类别的损失
    loss = torch.sum(margins) / N
    loss += reg * torch.sum(W * W)  #L2正则项

    # 计算梯度
    dscores = torch.zeros_like(scores)
    dscores[margins > 0] = 1
    dscores[torch.arange(N), y] -= torch.sum(dscores, dim=1)
    dW = X.t().mm(dscores) / N
    dW += 2 * reg * W
    return loss, dW


# 超参数搜索
learning_rates = [1e-3, 5e-3, 1e-2, 0.1]
reg_weights = [1e-4, 5e-4, 1e-3, 1e-2]
best_acc = 0
best_W = None
results = {}

for lr in learning_rates:
    for reg in reg_weights:
        loss = 0
        W = torch.randn(3072, 10, device=device) * 0.001
        for t in range(1000):
            loss, grad = svm_loss(X_train, y_train, W, reg)
            W -= lr * grad

        # 评估准确率
        scores_val = X_val.mm(W)
        y_val_pred = torch.argmax(scores_val, dim=1)
        val_acc = (y_val_pred == y_val).float().mean().item()

        results[(lr, reg)] = (loss, val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_W = W

# 使用最佳超参数训练模型
W = torch.randn(3072, 10, device=device) * 0.001
best_lr, best_reg = max(results, key=lambda k: results[k][1])
print(best_lr, best_reg)
for t in range(1000):
    loss, grad = svm_loss(X_train, y_train, W, best_reg)
    W -= best_lr * grad

    if t % 100 == 0:
        print(f'Iteration{t}, loss = {loss}')


# 评估测试集准确率
X_test = []
y_test = []
for images, labels in test_loader:
    X_test.append(images.view(images.size(0), -1))
    y_test.append(labels)
X_test = torch.cat(X_test, dim=0).to(device)
y_test = torch.cat(y_test, dim=0).to(device)

scores_test = X_test.mm(W)
y_test_pred = torch.argmax(scores_test, dim=1)
test_acc = (y_test_pred == y_test).float().mean().item()
print('Test accuracy:', test_acc)

def visualize_weights(W):
    W = W.cpu().numpy()  # 将权重矩阵从 GPU 移动到 CPU 并转换为 NumPy 数组
    W = W.reshape(10, 32, 32, 3)  # 重塑权重矩阵为 (10, 32, 32, 3)
    W_min, W_max = np.min(W), np.max(W)  # 计算权重矩阵的最小值和最大值
    W = (W - W_min) / (W_max - W_min)  # 归一化权重矩阵到 [0, 1] 范围
    W = (W * 255).astype(np.uint8)  # 将归一化后的权重矩阵转换为 8 位无符号整数

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 创建一个 2x5 的子图网格
    for i in range(10):
        ax = axes[i // 5, i % 5]  # 选择当前类别的子图
        ax.imshow(W[i])  # 在当前子图中显示第 i 类别的权重图像
        ax.set_title(f'Class {i}')  # 设置当前子图的标题
        ax.axis('off')  # 关闭当前子图的坐标轴
    plt.show()  # 显示整个图形

visualize_weights(W)



