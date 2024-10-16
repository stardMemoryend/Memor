# GPU运行，使用pytorch张量计算
import torch
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data
# 检查GPU
device = torch.device("cuda")

# 先加载数据集（训练集和测试集）， 然后将图像转换为张量
# 再将图像数据展平为一维向量， 接着再把数据转为数组，

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)) #将图像展平为一维向量
])

# train_dataset是一个CIFAR10对象，内部管理着图像数据和标签数据
# 可以像访问列表一样访问train_dataset, 返回元组(image, label)
# 训练集构建
train_dataset = datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
# 创建batch
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试集构建
test_dataset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

train_images = []
train_labels = []
test_images = []
test_labels = []

# 将数据集移动到GPU并用一个张量统一存储
for image, label in train_loader:
    image = image.to(device)
    label = label.to(device)
    # image: (64, 3072)
    train_images.append(image)
    # label: (64, 1)
    train_labels.append(label)

for image, label in test_loader:
    image = image.to(device)
    label = label.to(device)
    test_images.append(image)
    test_labels.append(label)

# 将所有图像数据连接到一个数组
# [ [1, 2, 3]
#   [4, 5, 6] ]  与
# [ [7, 7, 7]
#   [8, 8, 8] ]  合并成
# [ [1, 2, 3]
#   [4, 5, 6]
#   [7, 7, 7]
#   [8, 8, 8] ]
train_images = torch.cat(train_images)
train_labels = torch.cat(train_labels)
test_images = torch.cat(test_images)
test_labels = torch.cat(test_labels)
# 欧氏距离
def distance(x1, x2):
    return torch.cdist(x1, x2)

# KNN
def knn(X_train, y_train, X_test, k):
    predictions = []
    for test_batch in X_test.split(64):
        distances = distance(test_batch, X_train) #计算批量距离
        k_indices = torch.topk(distances, k, largest=False, dim=1).indices
        k_nearest_labels = y_train[k_indices]
        most_common_label, _ = torch.mode(k_nearest_labels, dim=1)
        # 预测标签
        predictions.append(most_common_label)
    return torch.cat(predictions)
# k折交叉验证
def cross_validate(X, y, k_folds, k_neighbors):
    fold_size = len(X) // k_folds  # 每个子集包含的样本数
    accuracies = []

    for fold in range(k_folds):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]   # 测试集
        X_train = torch.cat((X[:start], X[end:]))   # 训练集
        y_train = torch.cat((y[:start], y[end:]))

        # 预测验证集标签
        predicted_labels = knn(X_train, y_train, X_val, k_neighbors)

        # 准确率
        accuracy = (predicted_labels == y_val).float().mean()
        accuracies.append(accuracy)

    mean_accuracy = torch.tensor(accuracies).mean()
    return mean_accuracy

# 预测测试集标签
# predicted_labels = knn(train_images, train_labels, test_images, k=4)


# 计算准确率
# np.mean用于计算数组的平均值，predicted_labels == test_labels会得到布尔值0和1
# True为1，False为0
# accuracy = (predicted_labels == test_labels).float().mean()
# print(accuracy)

# 定义可能的超参数
k_values = [1, 2, 3, 4, 5]

# 网格搜索
best_k = None
best_accuracy = 0.0

for k in k_values:
    accuracy = cross_validate(train_images, train_labels, 5, k)
    print(f'K={k}, accuracy: {accuracy}')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print('\n')
print(best_k, best_accuracy)

pred = knn(train_images, train_labels, test_images, best_k)

true_accuracy = (pred == test_labels).float().mean()
print(true_accuracy)
