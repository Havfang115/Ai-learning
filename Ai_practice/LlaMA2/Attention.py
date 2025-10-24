import torch
import torch.nn as nn
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

