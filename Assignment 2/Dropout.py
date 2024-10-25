import torch
import torchvision.transforms as transforms
import torchvision.datasets
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

############## 手动实现dropout #################
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = dropout
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, is_training=True):
        out1 = self.fc1(x)
        out2 = self.bn1(out1)
        out3 = self.relu(out2)

        if is_training: # 训练时作用dropout,否则不作用
            # 生成随机掩码
            mask = (torch.rand_like(out2) > self.dropout).float()
            # 应用掩码并缩放
            out3 = out3 * mask / (1 - self.dropout)

        out4 = self.fc2(out3)

        return out4



############ 库函数实现dropout #############
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.bn1(out1)
        out3 = self.relu(out2)
        out3 = self.dropout(out3)
        out4 = self.fc2(out3)
        return out4


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=False, transform=transforms)
test_dataset = torchvision.datasets.CIFAR10('../data', train=False, download=False, transform=transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda")

net = TwoLayerNet(3072, 100, 10).to(device)
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.001)


def transfer(loader):
    X = []
    y = []
    for images, labels in loader:
        X.append(images.view(images.size(0), -1))
        y.append(labels)
    X = torch.cat(X, dim=0).to(device)
    y = torch.cat(y, dim=0).to(device)
    return X, y

X_train, y_train = transfer(train_loader)
X_test, y_test = transfer(test_loader)

num_epochs = 10
for epoch in range(num_epochs):
    net.train()  # 调整为训练模式
    for batch_idx in range(0, len(X_train), 100):
        X_batch = X_train[batch_idx:batch_idx + 100]
        y_batch = y_train[batch_idx:batch_idx + 100]
        # 前向传播
        outputs = net(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播
        optimizer.zero_grad()   #  梯度清零
        loss.backward()
        optimizer.step()

        if (batch_idx + 100) % 10000 == 0:
            print(f'Epoch: {epoch}, Step: {(batch_idx+100) / 100}, Loss: {loss}')

    net.eval()  # 调整为验证模式
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx in range(0, len(X_train), 100):
            X_batch = X_train[batch_idx:batch_idx + 100]
            y_batch = y_train[batch_idx:batch_idx + 100]
            outputs = net(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        accuracy = correct / total
        print(accuracy)


# 在测试集上进行最终评测
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_idx in range(0, len(X_test), 100):
        X_batch = X_test[batch_idx:batch_idx + 100]
        y_batch = y_test[batch_idx:batch_idx + 100]
        outputs = net(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    print(accuracy)






