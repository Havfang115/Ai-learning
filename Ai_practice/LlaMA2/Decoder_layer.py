import torch
import torch.nn as nn
from Model_Config import ModelConfig
from Attention import Attention
from MLP import MLP
from RMSNorm import RMSNorm
from Attention import precompute_freqs_cis

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()

        self.n_heads = args.n_heads # 定义多头注意力的头数
        self.dim = args.dim # 定义输入的维度
        self.head_dim = self.dim // self.n_heads # 定义每个注意力头的维度
        self.attention = Attention(args) # 定义注意力层
        # 定义LLaMAMLP层
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 定义层ID
        self.layer_id = layer_id
        # 定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义FFN计算的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    # 定义解码器层的前向传播过程
    def forward(self, x, freqs_cos, freqs_sin):
        # 首先，输入通过注意力归一化层，然后通过注意力层，结果与输入相加得到h
        # 然后，h通过FFN归一化层，然后通过FFN层，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
### 测试代码

args = ModelConfig()

# 创建LLaMADecoderLayer实例
decoderlayer = DecoderLayer(0, args)
# 模拟输入数据
dim = args.dim
seq_len = 50
# 定义输入数据
x = torch.randn(1, seq_len, dim) # [bs, seq_len, dim]
freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)
out = decoderlayer(x, freqs_cos, freqs_sin)

print(out.shape) # 形状和输入的x一样 [batch_size, seq_len, dim]