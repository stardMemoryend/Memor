# import os
# import json
# import pickle
#
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import models, transforms
#
# class CocoDataset(Dataset):
#     def __init__(self, root, annotations_file, word_to_idx, transform=None):
#         with open(annotations_file, 'r') as f:
#             self.annotations = json.load(f)['annotations']
#         self.root = root    # 图片根目录
#         self.transform = transform
#         self.word_to_idx = word_to_idx
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, idx):
#         # 获取单个样本
#         annotation = self.annotations[idx]
#         img_id = annotation['image_id']
#         caption = annotation['caption']
#         img_path = os.path.join(self.root, f'COCO_train2014_{img_id:012d}.jpg')
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#
#         # 将单词转换为索引
#         tokens = caption.split()
#         caption_indices = [self.word_to_idx.get(token, self.word_to_idx['<unk>']) for token in tokens]
#         caption_indices = [self.word_to_idx['<start>']] + caption_indices + [self.word_to_idx['<end>']]
#
#         return image, caption_indices
#
# # 将不同长度的标题都填充到相同长度
# def collate_fn(batch):
#     images, captions = zip(*batch)
#     images = torch.stack(images, 0)
#
#     # 计算最长的标题长度
#     max_len = max(len(cap) for cap in captions)
#
#     # 填充标题到相同长度
#     padded_captions = []
#
#     for cap in captions:
#         padded_cap = cap + [word_to_idx['<pad>']] * (max_len - len(cap))
#         padded_captions.append(padded_cap)
#
#     padded_captions = torch.tensor(padded_captions, dtype=torch.long)
#
#     return images, padded_captions
#
# class ImageEncoder(nn.Module):
#     def __init__(self, embed_size):
#         super(ImageEncoder, self).__init__()
#         resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#         modules = list(resnet.children())[:-1]  # 移除最后一个全连接层
#         self.resnet = nn.Sequential(*modules)
#         self.linear = nn.Linear(resnet.fc.in_features, embed_size)  # 添加线性层
#         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01) # 添加批量归一化层
#
#     def forward(self, images):
#         with torch.no_grad():   # 冻结ResNet50的参数
#             features = self.resnet(images)
#         features = features.view(features.size(0), -1)   # 展平特征
#         features = self.bn(self.linear(features))   #线性变换和批量归一化
#         return features
#
# class RNNDecoder(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(RNNDecoder, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)   # 词嵌入
#         self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, vocab_size)
#
#     def forward(self, features, captions):
#         embeddings = self.embed(captions)   # 将单词索引转换为词向量
#         features = features.unsqueeze(1)
#         print("features: ", features.shape)
#         print("embeddings: ", embeddings.shape)
#         embeddings = torch.cat((features, embeddings), dim=1)  # 将图像特征和词向量拼接
#         hidden, _ = self.rnn(embeddings)    # 提高RNN层
#         outputs = self.linear(hidden)   # 通过输出层
#         return outputs
#
# def train(encoder, decoder, data_loader, criterion, optimizer, device):
#     encoder.train()
#     decoder.train()
#     total_loss = 0.0
#     for images, captions in data_loader:
#         images, captions = images.to(device), captions.to(device)
#         targets = captions[:, 1:]   # 去掉起始标记后的标题
#
#         optimizer.zero_grad()
#
#         # 先通过编码器
#         features = encoder(images)
#         # 然后通过解码器
#         outputs = decoder(features, captions[:, :-1])   # 前向传播，输入去掉结束标记的标题
#
#         loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(data_loader)
#
# def generate_caption(image, encoder, decoder, vocab, max_len=20):
#     encoder.eval()
#     decoder.eval()
#     with torch.no_grad():
#         features = encoder(image.unsqueeze(0).to(device))   # 编码图像
#         caption = []    # 生成的标题
#         input_word = torch.tensor([[vocab['<start>']]]).to(device)  # 起始标记
#         states = None
#         for i in range(max_len):
#             outputs, states = decoder(features, input_word, states)
#             predicted = outputs.argmax(dim=-1)  # 选择概率最大的单词
#             caption.append(predicted.item())
#             if predicted.item() == vocab['<end>']:  # 侦测到结束标记则停止生成
#                 break
#     return [vocab['idx_to_word'][idx] for idx in caption] # 将单词索引转换为单词
#
#
# with open('../data/vocab.pkl', 'rb') as f:
#     word_to_idx, idx_to_word = pickle.load(f)
# vocab = {'<start>': word_to_idx['<start>'], '<end>': word_to_idx['<end>'], 'idx_to_word': idx_to_word}
#
#
# transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# root = '../data/train2014'
# annotations_file = '../data/annotations/captions_train2014.json'
# dataset = CocoDataset(root, annotations_file, word_to_idx, transform=transforms)
# data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
#
# embed_size = 256    # 嵌入维度
# hidden_size = 512   # 隐藏层大小
# vocab_size = 10000  # 词汇表大小
# num_layers = 1  # RNN层数
# learning_rate = 0.001
# num_epochs = 10
#
#
# device = torch.device("cuda")
# image_encoder = ImageEncoder(embed_size).to(device) # 图像编码器
# rnn_decoder = RNNDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)    # 标题解码器
#
# criterion = nn.CrossEntropyLoss(ignore_index=0) # 忽略索引为0的标签
# optimizer = optim.Adam(list(image_encoder.parameters()) + list(rnn_decoder.parameters()), lr=learning_rate)
#
#
# for epoch in range(num_epochs):
#     loss = train(image_encoder, rnn_decoder, data_loader, criterion, optimizer, device)
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss:{loss}')



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output = output_size
        self.Wxh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_size, 1))
        self.why = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)
        self.by = nn.Parameter(torch.zeros(output_size, 1))

    def forward(self, inputs, h_prev):
        seq_length, batch_size = inputs.size()
        # h为隐藏状态
        h = torch.zeros(seq_length + 1, batch_size, self.hidden_size)
        # h的最后一个被初始化为上一步的隐藏状态
        h[-1] = h_prev
        # 初始化输出, 存储每个时间步的输出
        y = torch.zeros(seq_length, batch_size, self.output)

        for t in range(seq_length): # 对每个时间步
            h[t] = torch.tanh(self.Wxh @ inputs[t] + self.Whh @ h[t-1] + self.bh)
            y[t] = self.why @ h[t] + self.by

        return y, h[-2]

def train(model, train_data, val_data, seq_length, batch_size, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        h_prev = torch.zeros(model.hidden_size, 1)

        for i in range(0, len(train_data) - seq_length, seq_length):
            inputs = torch.tensor(train_data[i:i+seq_length], dtype=torch.float32).unsqueeze(1)
            targets = torch.tensor(train_data[i+1:seq_length+1], dtype=torch.long)

            # 前向传播
            outputs, h_prev = model(inputs, h_prev)
            outputs = outputs.view(-1, outputs.size)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()














#
#
