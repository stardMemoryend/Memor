import torch
import torchvision.transforms as transforms
import torchvision.datasets
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

############### 手动实现 ##############

# 初始化参数
def init_bn_params(input_size):
    gamma = torch.ones(input_size, requires_grad=True, device=device)  # 缩放参数
    beta = torch.zeros(input_size, requires_grad=True, device=device)  # 平移参数
    running_mean = torch.zeros(input_size, device=device)  # 运行均值
    running_var = torch.ones(input_size, device=device)  # 运行方差
    return gamma, beta, running_mean, running_var


def batch_norm(X, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.9, is_training=True):
    if is_training:
        # 计算当前batch的均值和方差
        batch_mean = X.mean(dim=0)
        batch_var = X.var(dim=0, unbiased=False)

        # 更新运行均值和方差
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var

        # 标准化
        X_normalized = (X - batch_mean) / torch.sqrt(batch_var + eps)

    else:
        # 使用运行均值和方差进行标准化
        X_normalized = (X - running_mean) / torch.sqrt(running_var + eps)

    #  缩放和平移
    out = gamma * X_normalized + beta

    return out, running_mean, running_var


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gamma, self.beta, self.running_mean, self.running_var = init_bn_params(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out1 = self.fc1(x)
        out2, self.running_mean, self.running_var = batch_norm(
            out1, self.gamma, self.beta, self.running_mean, self.running_var,
            is_training=self.training
        )
        out3 = self.relu(out2)
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




############ 库函数实现 ###############

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # 批量归一化层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.bn1(out1)
        out3 = self.relu(out2)
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





