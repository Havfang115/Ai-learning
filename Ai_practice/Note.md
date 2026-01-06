# AI Learning Notes

## Overviews

Transformers：

- Encoders only: BERT. Understanding of text.
- Decoders only: GPT, Llama. Generation of text.
- Encoder-decoder: T5, BART. Sequence-to-sequence tasks.

Inference Process：(How trained model generates text)

- Prefill Phase: Tokenization, Embedding Conversion, Inital Processing.
- Decoding Phase: Attention, Probablity Calculation, Token Selection, Continuation Check.

Practical training:

- Pre-training -> Fine-tuning
- Transfer learning

How to trian a transformer:

- Masked Language Modeling (MLM): "Apple is [MASK]."
- Casual Language Modeling (CLM): From previous predict next.

Tokenizer:

- word-based
- character-based
- sub-word-based
- [CLS] at the beginning and [SEP] at the end of the sequence.

Key terms:

- Architecture: The structure of the model.
- Checkpoints: The saved weights of the model during training.
- Logits: The output of the model before the softmax function.
- Temperature: A hyperparameter that controls the randomness of the predictions.(Lower temperature = more randomness)
- Top-p sampling: Sampling only the top p%(probability threshold) of the logits.
- Zero-shot performance: The performance of the model on a task without any fine-tuning.

Key performance metrics:

- Time to First Token(TFT): The time taken to generate the first token.
- Time Per Output Token(TPOT): The time taken to generate each subsequent token.
- Throughput: The number of tokens generated per second.
- VRAM usage: The amount of memory used by the model.
- Perplexity: A measure of how well the model predicts the next token based on the probability distribution of the previous tokens.

Technical explanations:

- Multi-head attention: It works by dividing the input into multiple heads, each focusing on different parts of the input. The output of each head is then concatenated and passed through a linear layer to produce the final output.
- Self-attention: It is a type of attention that allows each position in the sequence to attend to all other positions in the sequence.
- Positional encoding: It is a vector that is added to the input embeddings to give the model information about the position of the word in the sentence. It helps the model to understand the order of the words in the sentence.
- Dropout: It is a regularization technique that randomly drops out some of the neurons during training to prevent overfitting.
- Beam search: It is a decoding technique that generates the most probable sequence of words by keeping track of the top k sequences at each step.
- KV Cache Optimization: It is a technique that imporves the inference speed of the transformer model by storing and reusing the intermediate calculations. The trade-off is that it requires more memory.

Python knowledge:

- output = model(**tokens): `**` is used to pass the dictionary as keyword arguments.

Characteristics of healthy curves:

- Smooth decline in loss: Both training and validation loss decrease steadily
- Close training/validation performance: Small gap between training and validation metrics
- Convergence: Curves level off, indicating the model has learned the patterns
