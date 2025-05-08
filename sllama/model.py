import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
@dataclass
class Config():
    dim: int = 4096
    n_embd: int = 768
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
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    def forward(self, x):
        B, T, C = x.size() # batch size(批量大小), Time(序列长度), dimension(向量维度)
        # split方法表示将dim维度分成每块为n_embd大小的块
        # q, k, v分别就是(B, T, C)维度的Tensor
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        att = q @ k.transpose(1,2) / torch.sqrt(self.n_embd)
        att = F.softmax(att, dim=-1)
        # att = (B, T, C)
        att = att @ v
        return att
attention = Attention()
