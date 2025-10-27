import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from Model_Config import ModelConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from Decoder_layer import DecoderLayer
from RMSNorm import RMSNorm
from Attention import precompute_freqs_cis

class Transformer(PreTrainedModel):
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
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args)) # 添加解码器层
        
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # 初始化最后一次向前传播的损失
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast() # 初始化输出结构
        self._no_split_modules = [name for name, _ in self.named_modules()] # 记录不需要进行切分的模块名称
    
    def _init_weights(self, module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # tokens: Optional[torch.Tensor] = None, 输入tokens序列
        # targets: Optional[torch.Tensor] = None, 目标序列，用于计算损失

        # 输入tokens序列
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']
        
        _bsz, seqlen = tokens.shape # 获取batch_size和序列长度
        h = self.tok_embeddings(tokens) # 词嵌入
        h = self.dropout(h) # dropout
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过decoder
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin) # 解码器层
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果targets不为空，则计算损失
            logits = self.output(h) # 计算logits
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # 展平logits
                targets.view(-1), # 展平targets
                ignore_index=0, # 忽略padding的位置
                reduction="None", # 无reduction，返回每个token的损失
            )
        else:
            # 如果targets为空，则返回预测结果
            # 只对最后一个位置的输出进行前向传播
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT
    
    @torch.inference_mode() # 推理模式, 关闭dropout等
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=0):
        # idx: 初始输入序列
        # stop_id: 停止生成的token id
        # max_new_tokens: 最大生成长度
        # temperature: 采样温度
        # top_k: top-k采样
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """

        index = idx.shape[1] # 当前序列长度
        for _ in range(max_new_tokens):
            # 如果当前序列长度超过最大序列长度，则停止生成
            idx_cond = idx if idx.shape(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len :]
            # 前向传播获取序列最后一个位置的logits
            logits = self(idx_cond).logits
            logits = logits[:, -1, :]

            if temperature == 0.0:
                # 选择概率最高的token
                _, idx_next = torch.topk(logits, k=1, dim=-1) # 取最大值
            else:
                # 缩放logits并应用softmax
                logits = logits / temperature # 调整温度
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # 取top_k最大值
                    logits[logits < v[:, [-1]]] = -float("Inf") # 将非top_k位置设为负无穷
                probs = F.softmax(logits, dim=-1) # 计算概率分布
                idx_next = torch.multinomial(probs, num_samples=1) # 根据概率分布采样

            if idx_next == stop_id: # 如果遇到停止token，则停止生成
                break

            # 将新生成的token拼接到输入序列中
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx[:, index:] # 只返回新生成的部分


### 测试代码

args = ModelConfig()

# LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
# 实例化LLaMA2Model
model = Transformer(args=args)
# 计算model的全部参数
num_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:', num_params)

out = model(x)
print(out.logits.shape) # [batch_size, 1, vocab_size]