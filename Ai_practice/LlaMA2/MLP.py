import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_Config import ModelConfig

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层维度，则将其设置为模型维度的4倍
        # 之后将其减少到2/3，之后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int((hidden_dim * 2) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义第一个线性层，输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二个线性层，隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性层，输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)

    # 前向传播
    def forward(self, x):
        # 首先通过第一个线性层和SILU激活函数
        # 之后乘以通过第三个线性层的输出
        # 最后通过第二个线性层并应用dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    

### 测试代码

args = ModelConfig()

# 创建MLP实例
mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
# 随机生成数据
x = torch.randn(1, 50, args.dim)
# 运行MLP模型
output = mlp(x)
print(output.shape)