# Transformer Model

Implementation of a Transformer model using PyTorch. The Transformer model is a deep learning architecture introduced in the paper "Attention is All You Need" by Vaswani et al. It is widely used for tasks involving sequence-to-sequence transformations, such as machine translation.

## Components

### SelfAttention

- **Purpose**: Computes attention scores for input sequences, allowing the model to focus on different parts of the input.
- **Key Operations**: Linear transformations of input queries, keys, and values, followed by scaled dot-product attention.

### TransformerBlock

- **Purpose**: A single block of the Transformer model, consisting of a self-attention layer followed by a feed-forward neural network.
- **Key Operations**: Layer normalization and dropout are applied to stabilize and regularize the training process.

### Encoder

- **Purpose**: Encodes the input sequence into a continuous representation.
- **Key Operations**: Embedding of input tokens and positional encoding, followed by multiple Transformer blocks.

### DecoderBlock

- **Purpose**: A single block of the Decoder, which attends to both the input sequence and the previously generated output.
- **Key Operations**: Self-attention for the target sequence, cross-attention with the encoded input, and a feed-forward network.

### Decoder

- **Purpose**: Decodes the encoded representation into the target sequence.
- **Key Operations**: Embedding of target tokens and positional encoding, followed by multiple Decoder blocks.

### Transformer

- **Purpose**: The complete Transformer model, combining the Encoder and Decoder.
- **Key Operations**: Generates masks for padding and future tokens, processes input through the Encoder and Decoder.

## Usage

The model can be instantiated and used for sequence-to-sequence tasks. Below is an example of how to initialize and run the model:

```python
import torch
from model import Transformer

device = torch.device("cpu")

# Example input and target sequences
x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

# Model parameters
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10

# Initialize the Transformer model
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

# Forward pass
out = model(x, trg[:, :-1])
print(out.shape)
```

## Requirements

- PyTorch
- Python 3.x

## Notes

- The model is designed to be flexible and can be adapted for various sequence-to-sequence tasks by adjusting the vocabulary size, embedding size, number of layers, and other hyperparameters.
- Ensure that the input and target sequences are appropriately padded and masked to handle variable-length sequences.
