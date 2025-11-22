Transformers：
- Encoders only: BERT. Understanding of text.
- Decoders only: GPT, Llama. Generation of text.
- Encoder-decoder: T5, BART. Sequence-to-sequence tasks.

Practical training:
- Pre-training -> Fine-tuning
- Transfer learning

How to trian a transformer:
- Masked Language Modeling (MLM): "Apple is [MASK]."
- Casual Language Modeling (CLM): From previous predict next.



Key terms:
- Architecture: The structure of the model.
- Checkpoints: The saved weights of the model during training.
- Logits: The output of the model before the softmax function.
- Zero-shot performance: The performance of the model on a task without any fine-tuning.

Technical explanations:
- Multi-head attention: It works by dividing the input into multiple heads, each focusing on different parts of the input. The output of each head is then concatenated and passed through a linear layer to produce the final output.
- Self-attention: It is a type of attention that allows each position in the sequence to attend to all other positions in the sequence.
- Positional encoding: It is a vector that is added to the input embeddings to give the model information about the position of the word in the sentence. It helps the model to understand the order of the words in the sentence.
- Dropout: It is a regularization technique that randomly drops out some of the neurons during training to prevent overfitting.
- 
