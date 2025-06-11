import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import time
import math


@dataclass
class Config:
    block_size: int = 3
    vocab_size: int = 50257 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 2
    n_embd: int = 8
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
class RMSNorm():
    def __init__(self):
        super().__init__()

class Attention(nn.Module):
    '''
    先实现Single attention，再实现multi-head attention
    搞清楚attention有多少层

    '''
    def __init__(self, config):
        super().__init__()
        # 加载config里面的参数
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    def forward(self, x):
        B, T, C = x.size() # batch size(批量大小), Time(序列长度), dimension(向量维度)
        # split方法表示将dim维度分成每块为n_embd大小的块
        # q, k, v分别就是(B, T, C)维度的Tensor
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        # 对c这个维度进行划分，让每个头专注于某一个这个单词的每一个特征
        # 不对B划分是因为每个头只能看见同一批的某一些样本
        # 不对T划分是因为每个头看不到全部的上下文信息，句子会被截断
        # qkv的维度经过multi-head之后为(B, T, nh, hs)
        # transpose之后的维度变为(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # k.transpose = (B, C, T)
        # q @ k.transpose = (B, nh, T, hs) @ (B, nh, hs, T) => (B, nh, T, T)
        att = q @ k.transpose(-1,-2) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # y = (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

