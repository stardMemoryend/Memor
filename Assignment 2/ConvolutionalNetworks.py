import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets
import torch.utils.data
import matplotlib.pyplot as plt

############ 第一种写法 ################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 初始图形尺寸：32*32
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # 尺寸：32*32
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)    # 尺寸：16*16
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 尺寸：16*16
        # 再调用池化层，尺寸：8*8， 通道数：32
        # 定义全连接层
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)  # 故张量尺寸：32*8*8
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        out1 = self.pool(F.relu(self.conv1(x)))
        out2 = self.pool(F.relu(self.conv2(out1)))
        # 展平向量
        out2 = out2.view(-1, 32*8*8)
        out3 = F.relu(self.fc1(out2))
        out4 = self.fc2(out3)
        return out4



############ 第二种写法 ###############
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.out = nn.Sequential(
            nn.Linear(in_features=32 * 8 * 8, out_features=128),
            nn.Linear(in_features=128, out_features=10)
        )
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        # out2 = out2.view(-1, 32*8*8)
        out2 = out2.view(out2.size(0), -1)  # out2.size(0)获取批量大小，-1自动计算维度大小
        out3 = self.out(out2)
        return out3

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])


train_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=False, transform=transforms)
test_dataset = torchvision.datasets.CIFAR10('../data', train=False, download=False, transform=transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda")
net = CNN().to(device)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.001)

def transfer(loader):
    X = []
    y = []
    for images, labels in loader:
        X.append(images)
        y.append(labels)
    X = torch.cat(X, dim=0).to(device)
    y = torch.cat(y, dim=0).to(device)
    return X, y

X_train, y_train = transfer(train_loader)
X_test, y_test = transfer(test_loader)

num_epochs = 50
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    net.train()  # 调整为训练模式
    train_loss = 0
    for batch_idx in range(0, len(X_train), 100):
        X_batch = X_train[batch_idx:batch_idx + 100]
        y_batch = y_train[batch_idx:batch_idx + 100]
        # 前向传播
        outputs = net(X_batch)
        train_loss = criterion(outputs, y_batch)

        # 反向传播
        optimizer.zero_grad()   #  梯度清零
        train_loss.backward()
        optimizer.step()

        if (batch_idx + 100) % 10000 == 0:
            print(f'Epoch: {epoch}, Step: {(batch_idx+100) / 100}, Loss: {train_loss}')
    train_losses.append(train_loss.item())

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

    # 记录测试集上损失函数的变化情况
    with torch.no_grad():
        for batch_idx in range(0, len(X_test), 100):
            X_batch = X_train[batch_idx:batch_idx + 100]
            y_batch = y_train[batch_idx:batch_idx + 100]
            outputs = net(X_batch)
            test_loss = criterion(outputs, y_batch)
        test_losses.append(test_loss.item())

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

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='train_loss', color='blue')
plt.plot(range(1, num_epochs + 1), test_losses, label='test_loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.show()

