# SmaLLM

A tiny language model written and trained from scratch, for enjoyment purposes.

## Overview

This project demonstrates the complete implementation of a Small Language Model (SLM) built entirely from first principles. The implementation includes a 60-million parameter transformer model capable of generating creative and coherent text, featuring custom attention mechanisms, positional encodings, and advanced training techniques including mixed precision, gradient accumulation, and learning rate scheduling.

## Technical Specifications

- **Architecture**: Transformer-based language model
- **Parameters**: ~60 million
- **Layers**: 6 transformer blocks
- **Attention heads**: 6 heads per layer
- **Embedding dimension**: 384
- **Context length**: 128 tokens
- **Vocabulary size**: 50,257 tokens (GPT-2 tokenizer)

## Dataset

The model is trained on the **TinyStories** dataset, a synthetic collection of short stories containing vocabulary that 3-4 year olds typically understand. This dataset was generated using GPT-3.5 and GPT-4, making it perfect for training smaller models while maintaining narrative coherence.

**TinyStories?**

- Manageable vocabulary size reduces model complexity
- Coherent narratives help the model learn story patterns
- Appropriate scale (~2M examples) for experimentation
- Clean, pre-processed data

## Implementation Details

### 1. Data Preprocessing

- **Tokenization**: Uses tiktoken (GPT-2 tokenizer) for efficient text-to-token conversion
- **Storage**: Pre-tokenized data stored as memory-mapped binary files (`train.bin`, `validation.bin`)
- **Memory efficiency**: Disk-based storage allows handling datasets larger than available RAM

### 2. Model Architecture

- **LayerNorm**: Custom normalization for stable training
- **CausalSelfAttention**: Multi-head attention with causal masking and Flash Attention support
- **MLP**: Position-wise feed-forward network with GELU activation
- **TransformerBlock**: Combines attention and MLP with residual connections
- **Weight tying**: Shared embeddings between input and output layers

### 3. Training Configuration

- **Optimizer**: AdamW with weight decay (0.1) and beta2=0.95
- **Learning rate**: 1e-4 with linear warmup (1000 steps) and cosine annealing
- **Batch size**: 32 with gradient accumulation (32 steps)
- **Training steps**: 20,000 iterations
- **Mixed precision**: bfloat16/float16 for faster training
- **Gradient clipping**: Max norm 0.5 for stability

### Training the Model

1. **Load and preprocess data** (automatically handles TinyStories dataset):

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
# Preprocessing creates train.bin and validation.bin files
```

2. **Model Training**:

```python
# Model configuration
config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

model = GPT(config)
# Training loop runs for 20,000 iterations
```

### Text Generation

```python
# Load trained model
model = GPT(config)
model.load_state_dict(torch.load('best_model_params.pt'))

# Generate text
sentence = "Once upon a time there was a pumpkin."
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0)
generated = model.generate(context, max_new_tokens=200)
print(enc.decode(generated.squeeze().tolist()))
```


## Technical Stack

- **PyTorch**: Deep learning framework
- **tiktoken**: Tokenization (GPT-2 tokenizer)
- **datasets**: HuggingFace datasets library
- **numpy**: Numerical computing
- **matplotlib**: Visualization
- **tqdm**: Progress tracking
