import torch
import torch.nn as nn
from typing import Optional
from Model_Config import ModelConfig
from transformers import PretrainedModel

class Transformer(PretrainedModel):
    config_class = ModelConfig # 指定配置类为ModelConfig
    last_loss: Optional[torch.Tensor] # 记录上一次的loss，用于判断是否需要进行梯度累积

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        self.args = args # 初始化配置参数
        self.vocab_size = args.vocab_size # 词汇表大小
        self.n_layers = args.n_layers # Transformer层数

        # 初始化模型层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim) # 词嵌入层
        self.dropout = nn.Dropout(args.dropout) # 随机失活层
        self.layers = torch.nn.ModuleList() # 模型层列表
        