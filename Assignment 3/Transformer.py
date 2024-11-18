import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # 存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 表示每个位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算位置编码的频率
        pe[:, 0::2] = torch.sin(position * div_term)    # 计算位置编码的正弦部分
        pe[:, 1::2] = torch.cos(position * div_term)    # 计算位置编码的余弦部分
        pe = pe.unsqueeze(0).transpose(0, 1)    # 将pe形状改为(max_len, 1, d_model)
        self.register_buffer('pe', pe)  # 将pe注册为缓冲区，不会被视为模型参数，但会被保存和加载

    # 实现位置编码的添加
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# 多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads # 计算每个头的维度
        self.num_heads = num_heads
        # 创建四个线性层，分别用于查询(Q)，键（K），值（V），以及最终的输出投影
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 如果有掩码，则对掩码为0的位置对应的注意力分数设置为1e-9,相当于忽略权重
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算权重
        attn = torch.softmax(scores, dim=-1)
        # 乘值value
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 线性投影，得到形状为（batch_size, seq_len, d_model)的输出
        return self.linears[-1](output)

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        attn_output = self.self_attn(src, src, src, src_mask)
        # 残差连接
        src = src + self.dropout1(attn_output)
        # 归一化
        src = self.norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src

# 编码器
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.src_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        attn_output = self.src_attn(tgt, memory, memory, src_mask)
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)

        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        return tgt

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, src_mask)
        return self.norm(tgt)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.dropout = dropout

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src = self.positional_encoding(src_embed)
        tgt = self.positional_encoding(tgt_embed)

        memory = self.encoder(src, src_mask)

        output = self.decoder(tgt, memory, tgt_mask, src_mask)

        output = self.fc_out(output)
        return output










