import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data

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
X_test = []
y_test = []

def transfer(X, y, loader):
    for images, labels in loader:
        X.append(images.view(images.size(0), -1))
        y.append(labels)
    X = torch.cat(X, dim=0).to(device)
    y = torch.cat(y, dim=0).to(device)
    return X, y

X_train, y_train = transfer(X_train, y_train, train_loader)
X_test, y_test = transfer(X_test, y_test, test_loader)

def softmax_loss(W, X, y, reg):
    # 计算分数
    scores = X.mm(W)
    # 数值稳定性：从每行中减去最大的分数
    # 如果不减，可能会导致数值溢出：exp(z)非常大，超出浮点数的表示范围
    # 可能会导致数值下溢，exp(Z)非常接近于零，无法精确表示接近于零的数
    # keepdim=True保持原维度，变为(N,1),若不进行则为(N),max返回的是元组(max_values, max_indices)，再[0]即取最大值
    # 然后由广播机制进行相减
    scores = scores - torch.max(scores, dim=1, keepdim=True)[0]

    # 计算概率
    exp_scores = torch.exp(scores)
    # sum返回的是张量(sum_values)
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

    # 计算损失
    N = X.shape[0]
    correct_prob = -torch.log(probs[torch.arange(N), y])
    loss = torch.sum(correct_prob) / N
    # 正则项
    loss += reg * torch.sum(W * W)

    # 计算梯度
    dscores = probs
    dscores[torch.arange(N), y] -= 1
    dscores /= N
    dW = X.t().mm(dscores)
    dW += 2 * reg * W

    return loss, dW

k = 5

def cross_validate(X, y, reg, k_folds, lr):
    fold_sizes = len(X) // k_folds
    accuracies = []

    for fold in range(k_folds):
        start, end = fold * fold_sizes, (fold + 1) * fold_sizes
        X_val, y_val = X[start:end], y[start:end]   # 验证集
        X_train = torch.cat((X[:start], X[end:]))   # 训练集
        y_train = torch.cat((y[:start], y[end:]))

        # 初始化权重矩阵
        W = torch.randn(3072, 10, device=device) * 0.001
        for t in range(1000):
            loss, grad = softmax_loss(W, X_train, y_train, reg)
            W -= lr * grad

        # 评估准确率
        scores_val = X_val.mm(W)
        y_val_pred = torch.argmax(scores_val, dim=1)
        val_acc = (y_val_pred == y_val).float().mean().item()
        accuracies.append(val_acc)

    mean_accuracy = torch.tensor(accuracies).mean().item()
    return mean_accuracy

reg_weights = [1e-5, 1e-4, 1e-3]
learning_rates = [1e-3, 5e-3, 1e-2]
k_folds = 5

best_reg = 0
best_lr = 0
best_accuracy = 0.0

for lr in learning_rates:
    for reg in reg_weights:
        accuracy = cross_validate(X_train, y_train, reg, k_folds, lr)
        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            best_lr = lr
            best_reg = reg

print(best_lr, best_reg, best_accuracy)

# 使用最佳超参数训练权重
W = torch.randn(3072, 10, device=device) * 0.001
for t in range(1000):
    loss, grad = softmax_loss(W, X_train, y_train, best_reg)
    W -= best_lr * grad

    if t % 100 == 0:
        print(f'Iteration{t}, loss = {loss}')

# 评估测试集准确率
scores_test = X_test.mm(W)
y_test_pred = torch.argmax(scores_test, dim=1)
test_acc = (y_test_pred == y_test).float().mean().item()
print(f'best test_accuracy: {test_acc}')

def visualize_weights(W):
    W = W.cpu().numpy()
    W = W.reshape(10, 32, 32, 3)
    W_min, W_max = np.min(W), np.max(W)
    W = (W - W_min) / (W_max - W_min)
    W = (W * 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 创建一个 2x5 的子图网格
    for i in range(10):
        ax = axes[i // 5, i % 5]  # 选择当前类别的子图
        ax.imshow(W[i])  # 在当前子图中显示第 i 类别的权重图像
        ax.set_title(f'Class {i}')  # 设置当前子图的标题
        ax.axis('off')  # 关闭当前子图的坐标轴
    plt.show()  # 显示整个图形


visualize_weights(W)








