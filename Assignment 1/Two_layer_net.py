# 方式一、将整个数据集直接放到GPU上，然后对数据集中预处理，然后再将这一整块数据放到模型里跑
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=False, transform=transforms)
test_dataset = torchvision.datasets.CIFAR10('../data', train=False, download=False, transform=transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda")

def transfer(X, y, loader):
    for images, labels in loader:
        X.append(images.view(images.size(0), -1))
        y.append(labels)
    X = torch.cat(X, dim=0).to(device)
    y = torch.cat(y, dim=0).to(device)
    return X, y

X_train, y_train = transfer([], [], train_loader)
X_test, y_test = transfer([], [], test_loader)

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = torch.randn(input_size, hidden_size, device=device) * 0.001
        self.params['b1'] = torch.zeros(hidden_size, device=device)
        self.params['W2'] = torch.randn(hidden_size, output_size, device=device) * 0.001
        self.params['b2'] = torch.zeros(output_size, device=device)
        for i in self.params:
            self.params[i].requires_grad_(True)

    def reset_params(self):
        self.params['W1'] = torch.randn(self.params['W1'].shape, device=device) * 0.001
        self.params['b1'] = torch.zeros(self.params['b1'].shape, device=device)
        self.params['W2'] = torch.randn(self.params['W2'].shape, device=device) * 0.001
        self.params['b2'] = torch.zeros(self.params['b2'].shape, device=device)
        for i in self.params:
            self.params[i].requires_grad_(True)

    def loss(self, X, y, reg):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 前向传播
        h1 = X.mm(W1) + b1
        a1 = torch.relu(h1)
        scores = a1.mm(W2) + b2

        # softmax损失
        scores -= torch.max(scores, dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores)
        probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

        N = X.shape[0]
        correct_prob = -torch.log(probs[torch.arange(N), y])
        loss = torch.sum(correct_prob) / N
        # 正则项
        loss += reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

        # 反向传播
        dscores = probs
        dscores[torch.arange(N), y] -= 1
        dscores /= N

        dW2 = a1.t().mm(dscores)
        db2 = torch.sum(dscores, dim=0)
        da1 = dscores.mm(W2.t())
        dh1 = da1.clone()
        dh1[h1 <= 0] = 0

        dW1 = X.t().mm(dh1)
        db1 = torch.sum(dh1, dim=0)

        # 添加正则化梯度
        dW2 += 2 * reg * W2
        dW1 += 2 * reg * W1

        # 更新参数
        grads = {}
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads

    def predict(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 前向传播
        h1 = X.mm(W1) + b1
        a1 = torch.relu(h1)
        scores = a1.mm(W2) + b2
        y_pred = torch.argmax(scores, dim=1)
        return y_pred

    def cross_validate(self, X, y, reg, lr, k_folds):
        fold_sizes = len(X) // k_folds
        accuracies = []

        for fold in range(k_folds):
            start, end = fold * fold_sizes, (fold + 1) * fold_sizes
            X_val, y_val = X[start:end], y[start:end]  # 验证集
            X_train = torch.cat((X[:start], X[end:]))  # 训练集
            y_train = torch.cat((y[:start], y[end:]))

            self.reset_params()

            for t in range(1000):
                loss, grad = self.loss(X_train, y_train, reg)
                with torch.no_grad():
                    for param in self.params:
                        self.params[param] -= lr * grad[param]

            y_val_pred = self.predict(X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            accuracies.append(val_acc)

        mean_accuracy = torch.tensor(accuracies).mean().item()
        return mean_accuracy

    def train(self, X, y):
        reg_weights = [1e-5, 1e-4, 1e-3]
        learning_rates = [1e-3, 5e-3, 1e-2]
        k_folds = 5

        best_reg = 0
        best_lr = 0
        best_accuracy = 0.0

        for lr in learning_rates:
            for reg in reg_weights:
                accuracy = self.cross_validate(X, y, reg, lr, k_folds)
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    best_lr = lr
                    best_reg = reg

        print(best_lr, best_reg, best_accuracy)

        self.reset_params()
        losses = []
        for t in range(1000):
            loss, grad = self.loss(X, y, best_reg)
            with torch.no_grad():
                for param in self.params:
                    self.params[param] -= best_lr * grad[param]

            if t % 100 == 0:
                print(f'Iteration{t}, loss = {loss.item()}')
                losses.append(loss.item())

        plt.plot(range(0, 1000, 100), losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

    def evaluate(self, X, y):
        y_test_pred = self.predict(X)
        test_acc = (y_test_pred == y).float().mean().item()
        print(f'best test_accuracy: {test_acc}')

net =  TwoLayerNet(3072, 100, 10)
net.train(X_train, y_train)
net.evaluate(X_test, y_test)




# # 方式二：分批次batch将部分数据放到GPU上，然后在模型里运行
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设定设备
device = torch.device("cuda")

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = torch.randn(input_size, hidden_size, device=device) * 0.001
        self.params['b1'] = torch.zeros(hidden_size, device=device)
        self.params['W2'] = torch.randn(hidden_size, output_size, device=device) * 0.001
        self.params['b2'] = torch.zeros(output_size, device=device)
        for i in self.params:
            self.params[i].requires_grad_(True)

    def reset_params(self):
        self.params['W1'] = torch.randn(self.params['W1'].shape, device=device) * 0.001
        self.params['b1'] = torch.zeros(self.params['b1'].shape, device=device)
        self.params['W2'] = torch.randn(self.params['W2'].shape, device=device) * 0.001
        self.params['b2'] = torch.zeros(self.params['b2'].shape, device=device)
        for i in self.params:
            self.params[i].requires_grad_(True)

    def loss(self, X, y, reg):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 前向传播
        h1 = X.mm(W1) + b1
        a1 = torch.relu(h1)
        scores = a1.mm(W2) + b2

        # softmax损失
        scores -= torch.max(scores, dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores)
        probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

        N = X.shape[0]
        correct_prob = -torch.log(probs[torch.arange(N), y])
        loss = torch.sum(correct_prob) / N
        # 正则项
        loss += reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

        # 反向传播
        dscores = probs
        dscores[torch.arange(N), y] -= 1
        dscores /= N

        dW2 = a1.t().mm(dscores)
        db2 = torch.sum(dscores, dim=0)
        da1 = dscores.mm(W2.t())
        dh1 = da1.clone()
        dh1[h1 <= 0] = 0

        dW1 = X.t().mm(dh1)
        db1 = torch.sum(dh1, dim=0)

        # 添加正则化梯度
        dW2 += 2 * reg * W2
        dW1 += 2 * reg * W1

        # 更新参数
        grads = {}
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads

    def predict(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 前向传播
        h1 = X.mm(W1) + b1
        a1 = torch.relu(h1)
        scores = a1.mm(W2) + b2
        y_pred = torch.argmax(scores, dim=1)
        return y_pred

    def cross_validate(self, dataset, reg, lr, k_folds):
        fold_sizes = len(dataset) // k_folds
        accuracies = []

        for fold in range(k_folds):
            start, end = fold * fold_sizes, (fold + 1) * fold_sizes
            val_indices = list(range(start, end))
            train_indices = list(range(start)) + list(range(end, len(dataset)))

            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)

            train_loader = DataLoader(train_subset, batch_size=100, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=100, shuffle=False)

            self.reset_params()

            for t in range(10):
                print('1')
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

                    X_batch = X_batch.view(X_batch.size(0), -1).to(device)
                    y_batch = y_batch.to(device)
                    loss, grad = self.loss(X_batch, y_batch, reg)
                    with torch.no_grad():
                        for param in self.params:
                            self.params[param] -= lr * grad[param]

            val_acc = self.evaluate(val_loader)
            accuracies.append(val_acc)

        mean_accuracy = torch.tensor(accuracies).mean().item()
        return mean_accuracy

    def train(self, dataset):
        reg_weights = [1e-5, 1e-4, 1e-3]
        learning_rates = [1e-3, 5e-3, 1e-2]
        k_folds = 5

        best_reg = 0
        best_lr = 0
        best_accuracy = 0.0

        for lr in learning_rates:
            for reg in reg_weights:
                accuracy = self.cross_validate(dataset, reg, lr, k_folds)
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    best_lr = lr
                    best_reg = reg

        print(f'Best learning rate: {best_lr}, Best regularization: {best_reg}, Best accuracy: {best_accuracy}')

        self.reset_params()
        losses = []

        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

        for t in range(10):
            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                X_batch = X_batch.view(X_batch.size(0), -1).to(device)
                y_batch = y_batch.to(device)
                loss, grad = self.loss(X_batch, y_batch, best_reg)
                with torch.no_grad():
                    for param in self.params:
                        self.params[param] -= best_lr * grad[param]

                if batch_idx % 10 == 0:
                    print(f'Epoch {t}, Batch {batch_idx}, loss = {loss.item()}')
                    losses.append(loss.item())

        plt.plot(range(len(losses)), losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

    def evaluate(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                X = X.view(X.size(0), -1)
                y_pred = self.predict(X)
                correct += (y_pred == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        return test_acc

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=False, transform=transforms)
test_dataset = torchvision.datasets.CIFAR10('../data', train=False, download=False, transform=transforms)

net = TwoLayerNet(3072, 100, 10)
# 训练模型
net.train(train_dataset)

# 评估模型
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
test_acc = net.evaluate(test_loader)
print(f'accuracy: {test_acc}')





# 三、将整个数据集先放到GPU上，然后预处理，再在模型里分批次batch运行
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 数据预处理
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 加载数据集
train_dataset = datasets.CIFAR10('../data', train=True, download=False, transform=transforms)
test_dataset = datasets.CIFAR10('../data', train=False, download=False, transform=transforms)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda")

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

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = torch.randn(input_size, hidden_size, device=device) * 0.001
        self.params['b1'] = torch.zeros(hidden_size, device=device)
        self.params['W2'] = torch.randn(hidden_size, output_size, device=device) * 0.001
        self.params['b2'] = torch.zeros(output_size, device=device)
        for i in self.params:
            self.params[i].requires_grad_(True)

    def reset_params(self):
        self.params['W1'] = torch.randn(self.params['W1'].shape, device=device) * 0.001
        self.params['b1'] = torch.zeros(self.params['b1'].shape, device=device)
        self.params['W2'] = torch.randn(self.params['W2'].shape, device=device) * 0.001
        self.params['b2'] = torch.zeros(self.params['b2'].shape, device=device)
        for i in self.params:
            self.params[i].requires_grad_(True)

    def loss(self, X, y, reg):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 前向传播
        h1 = X.mm(W1) + b1
        a1 = torch.relu(h1)
        scores = a1.mm(W2) + b2

        # softmax损失
        scores -= torch.max(scores, dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores)
        probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

        N = X.shape[0]
        correct_prob = -torch.log(probs[torch.arange(N), y])
        loss = torch.sum(correct_prob) / N
        # 正则项
        loss += reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

        # 反向传播
        dscores = probs
        dscores[torch.arange(N), y] -= 1
        dscores /= N

        dW2 = a1.t().mm(dscores)
        db2 = torch.sum(dscores, dim=0)
        da1 = dscores.mm(W2.t())
        dh1 = da1.clone()
        dh1[h1 <= 0] = 0

        dW1 = X.t().mm(dh1)
        db1 = torch.sum(dh1, dim=0)

        # 添加正则化梯度
        dW2 += 2 * reg * W2
        dW1 += 2 * reg * W1

        # 更新参数
        grads = {}
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads

    def predict(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 前向传播
        h1 = X.mm(W1) + b1
        a1 = torch.relu(h1)
        scores = a1.mm(W2) + b2
        y_pred = torch.argmax(scores, dim=1)
        return y_pred

    def cross_validate(self, X, y, reg, lr, k_folds):
        fold_sizes = len(X) // k_folds
        accuracies = []

        for fold in range(k_folds):
            start, end = fold * fold_sizes, (fold + 1) * fold_sizes
            X_val, y_val = X[start:end], y[start:end]  # 验证集
            X_train = torch.cat((X[:start], X[end:]))  # 训练集
            y_train = torch.cat((y[:start], y[end:]))

            self.reset_params()

            for t in range(10):
                print('1')
                for batch_idx in range(0, len(X_train), 100):
                    X_batch = X_train[batch_idx:batch_idx + 100]
                    y_batch = y_train[batch_idx:batch_idx + 100]
                    loss, grad = self.loss(X_batch, y_batch, reg)
                    with torch.no_grad():
                        for param in self.params:
                            self.params[param] -= lr * grad[param]

            val_acc = self.evaluate(X_val, y_val)
            accuracies.append(val_acc)

        mean_accuracy = torch.tensor(accuracies).mean().item()
        return mean_accuracy

    def train(self, X_train, y_train):
        reg_weights = [1e-5, 1e-4, 1e-3]
        learning_rates = [1e-3, 5e-3, 1e-2]
        k_folds = 5

        best_reg = 0
        best_lr = 0
        best_accuracy = 0.0

        for lr in learning_rates:
            for reg in reg_weights:
                accuracy = self.cross_validate(X_train, y_train, reg, lr, k_folds)
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    best_lr = lr
                    best_reg = reg

        print(f'Best learning rate: {best_lr}, Best regularization: {best_reg}, Best accuracy: {best_accuracy}')

        self.reset_params()
        losses = []

        for t in range(10):
            for batch_idx in range(0, len(X_train), 100):
                X_batch = X_train[batch_idx:batch_idx + 100]
                y_batch = y_train[batch_idx:batch_idx + 100]
                loss, grad = self.loss(X_batch, y_batch, best_reg)
                with torch.no_grad():
                    for param in self.params:
                        self.params[param] -= best_lr * grad[param]

                if batch_idx % 1000 == 0:
                    print(f'Epoch {t}, Batch {batch_idx}, loss = {loss.item()}')
                    losses.append(loss.item())

        plt.plot(range(len(losses)), losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

    def evaluate(self, X, y):
        with torch.no_grad():
            y_pred = self.predict(X)
            test_acc = (y_pred == y).float().mean().item()
        return test_acc

net = TwoLayerNet(3072, 100, 10)
net.train(X_train, y_train)
test_acc = net.evaluate(X_test, y_test)

print(f'accuracy: {test_acc}')

