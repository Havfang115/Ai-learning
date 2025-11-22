from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased") # 创建一个情感分析管道
unmasker("I just went to the cinema the other day and it was [MASK]") # 输入句子，输出情感分析结果
