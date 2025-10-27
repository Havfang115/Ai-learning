import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from Model_Config import ModelConfig

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批次大小，序列长度，头数，头维度
    bs, slen, n_kv_heads, head_dim = x.shape

    # 如果重复次数为1，直接返回输入张量
    if n_rep == 1:
        return x

    # 重复键值张量
    return (
        x[:, :, :, None, :] # 扩展维度，准备重复
        .expand(bs, slen, n_kv_heads, n_rep, head_dim) # 扩展键值张量到重复次数
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # 重新排列维度，合并头数和重复次数
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 计算频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 计算角度
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    # 计算频率和角度的乘积
    freqs = torch.outer(t, freqs).float()
    # 计算频率和角度的乘积的余弦值
    freqs_cos = torch.cos(freqs)
    # 计算频率和角度的乘积的正弦值
    freqs_sin = torch.sin(freqs)
    # 返回频率和角度的乘积
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 计算输入张量的形状
    ndim = x.ndim
    # 检查输入张量的维度是否在有效范围内
    assert 0 <= 1 < ndim
    # 检查频率张量的形状是否与输入张量的形状匹配
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 构建广播形状
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 广播频率张量到查询张量的形状
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转嵌入到查询张量
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    # 应用旋转嵌入到键张量
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 合并实部和虚部
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # 根据是否指定n_kv_heads来设置键值对注意力头的数量
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保注意力头的数量可以被键值对注意力头的数量整除
        assert args.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        # 模型并行大小
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行大小
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值对头数除以模型并行大小
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，等于本地计算头数除以本地键值头数，用于扩展键值对张量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头维度， 等于模型维度除以总头数
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 定义输出矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout概率
        self.dropout = args.dropout

        # 检查是否使用Flash Attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            # 无法使用FA则手动实现注意力机制并设置mask
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1) # 上三角矩阵
            # 注册缓冲区以保存mask
            self.register_buffer("mask", mask)

    # 定义前向传播函数
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # 获取批次大小和序列长度，【batch_size, seq_len, dim】
        bsz, seqlen, _ = x.shape
        
        # 计算查询、键、值张量
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x) 
        # 调整形状以适应头的维度
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转嵌入(RoPE)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        
        # 重复键值对张量以匹配查询张量的头数
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为批次维度的一部分进行处理
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据是否支持Flash Attention选择计算方法
        if self.flash:
            # 使用Flash Attention
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, 
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=True
            )
        else:
            # 手动实现注意力机制
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)
        
        # 恢复时间维度并合并头
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 投影回残差连接维度
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
    
args = ModelConfig()
# 测试代码
attention = Attention(args)

# 模拟输入数据
batch_size = 1
seq_len = 50
dim = args.dim
x = torch.randn(batch_size, seq_len, dim) # 输入张量
freqs_cos, freqs_sin = precompute_freqs_cis(dim // args.n_heads, seq_len)
output = attention(x, freqs_cos, freqs_sin)
print("Attention output shape:", output.shape)  # 应该是 (1, 50, dim)


