# GPT transformer architecture from Scratch

A minimal implementation of a Generative Pre-trained Transformer (GPT) model built from scratch using PyTorch. This project demonstrates the core components of transformer architecture including self-attention mechanisms, multi-head attention, and the decoder-only architecture that powers modern language models.

![Self-Attention Illustration](img.png)

## Overview

This implementation includes a complete GPT model with:
- **Self-attention mechanism** with causal masking (decoder block)
- **Multi-head attention** for parallel attention computation
- **Feed-forward networks** with residual connections
- **Layer normalization** and dropout for regularization
- **Character-level tokenization** for text processing

## Architecture

The model consists of:
- **Token and Position Embeddings**: Maps characters to dense vectors and adds positional information
- **Transformer Blocks**: Stack of attention and feed-forward layers with residual connections
- **Language Model Head**: Linear layer that outputs logits over the vocabulary

### Model Components

- `Head`: Single self-attention head with causal masking
- `MultiHeadAttention`: Parallel attention heads with projection
- `FeedForward`: Two-layer MLP with ReLU activation
- `Block`: Transformer block combining attention and feed-forward with layer norm
- `GPT`: Complete model with embedding layers and stacked transformer blocks

## Files

- `gpt_model.py`: GPT model architecture implementation
- `gpt_train.py`: Training script with data loading, training loop, and text generation
- `input.txt`: Training dataset (text corpus)
- `gpt.pth`: Saved model weights (after training)
- `more.txt`: Generated text output

## Requirements

- Python 3.x
- PyTorch
- tqdm (for progress bars)

## Usage

1. **Prepare your dataset**: Place your text corpus in `input.txt` (or download the TinyShakespeare dataset as mentioned in the code)

2. **Train the model**:
   ```bash
   python gpt_train.py
   ```

3. **Model Configuration**: The training script includes configurable hyperparameters:
   - `batch_size`: Number of sequences processed in parallel (default: 64)
   - `block_size`: Maximum context length (default: 256)
   - `n_embd`: Embedding dimension (default: 384)
   - `n_layers`: Number of transformer blocks (default: 6)
   - `n_heads`: Number of attention heads (default: 6)
   - `dropout`: Dropout rate (default: 0.2)
   - `learning_rate`: Learning rate for AdamW optimizer (default: 3e-4)
   - `max_iters`: Maximum training iterations (default: 5000)

4. **Output**: After training, the model will:
   - Save weights to `gpt.pth`
   - Generate text and save it to `more.txt`

## Training Details

- Uses character-level tokenization (builds vocabulary from unique characters in the dataset)
- 90/10 train/validation split
- AdamW optimizer
- Evaluates loss on validation set every 500 iterations
- Automatically uses CUDA if available, otherwise falls back to CPU

## Model Size

The default configuration results in approximately 10M parameters.
