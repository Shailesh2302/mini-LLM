# mini-LLM

A complete implementation of a modern Large Language Model built from scratch in PyTorch, featuring state-of-the-art architectural components.

## 📋 Overview

**mini-LLM** implements a transformer-based language model with modern optimizations used in production LLMs like Llama 2 and Mistral. It's designed for educational purposes, demonstrating how contemporary language models work while maintaining code clarity.

The project trains on the tiny-shakespeare dataset, learning to generate character-level text.

## ✨ Key Features

### Modern Transformer Components

- **RMSNorm** - Root Mean Square Layer Normalization
  - Simpler and faster than standard LayerNorm
  - Skips mean subtraction, keeping only RMS scaling
  
- **Rotary Positional Embeddings (RoPE)** - Advanced position encoding
  - Rotates Q and K vectors based on position
  - Captures relative distances naturally
  - Used in Llama, Mistral, and other modern LLMs
  
- **Grouped Query Attention (GQA)** - Efficient attention mechanism
  - Reduces KV cache by grouping query heads
  - 4x-8x memory savings compared to standard multi-head attention
  - Same quality with lower computational cost
  
- **SwiGLU** - Gated feed-forward network
  - Two-path architecture: gate and value paths
  - Uses SiLU activation for smoother gradients
  - Better performance than traditional ReLU-based FFNs

### Training & Inference

- Character-level tokenization
- Full training loop with PyTorch
- Text generation capabilities
- Pre-computed RoPE frequency tables for efficiency

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- Matplotlib (for visualizations)

### Installation

```bash
git clone https://github.com/Shailesh2302/mini-LLM.git
cd mini-LLM
```

### Usage

The main notebook `LLM.ipynb` contains the complete implementation:

1. **Data Preparation** - Downloads and preprocesses tiny-shakespeare dataset
2. **Architecture** - Implements all components from scratch
3. **Training** - Full training pipeline with loss tracking
4. **Generation** - Sample text generation from trained model

Open and run `LLM.ipynb` in Jupyter:

```bash
jupyter notebook LLM.ipynb
```

## 📚 Project Structure

```
mini-LLM/
├── LLM.ipynb           # Complete implementation and training
├── README.md           # This file
└── LICENSE             # Apache License 2.0
```

## 🏗️ Architecture Details

### Model Components

**RMSNorm (Root Mean Square Normalization)**
```
output = (input / RMS(input)) * learnable_scale
```
Faster and simpler than LayerNorm while maintaining similar effectiveness.

**RoPE (Rotary Positional Embeddings)**
- Different rotation frequencies for different dimension pairs
- Low dimensions: fast rotation → short-range patterns
- High dimensions: slow rotation → long-range patterns
- Relative position awareness through rotation angle differences

**GQA (Grouped Query Attention)**
```
n_heads: 8 (Query heads)
n_kv_heads: 2 (Key/Value heads)
n_rep: 4 (KV heads repeated 4 times)
Result: 4x smaller KV cache
```

**SwiGLU (Feed-Forward Network)**
```
gate = SiLU(W_gate @ x)
up = W_up @ x
output = W_down @ (gate * up)
```

### TransformerBlock

Pre-norm architecture with residual connections:
```
1. Normalize → GQA Attention → Add Residual
2. Normalize → SwiGLU FFN → Add Residual
```

### MiniLLM

Complete model combining all components:
- Token embeddings (no separate positional embeddings - RoPE handles it)
- N transformer blocks
- Final RMSNorm + language model head
- Weight tying between embedding and output layers

## 🔧 Technologies

- **Language**: Python 3
- **Framework**: PyTorch
- **Notebook**: Jupyter Notebook
- **License**: Apache License 2.0

## 📖 Learning Path

The notebook is organized into sections:

0. **Setup** - Environment and dependencies
1. **RMSNorm** - Understanding efficient normalization
2. **RoPE** - Modern positional encodings with visualizations
3. **GQA** - Efficient attention mechanisms
4. **SwiGLU** - Advanced feed-forward networks
5. **Full Model Assembly** - Putting it all together
6. **Training** - Complete training pipeline
7. **Generation** - Text generation from trained model

## 🎯 Why These Components Matter

Modern LLMs like Llama 2, Mistral, and Qwen use all these components because they provide:

- **Better Training Efficiency** - RMSNorm and SwiGLU improve gradient flow
- **Better Inference** - RoPE and GQA reduce memory and computational requirements
- **Better Scaling** - These techniques maintain quality while reducing resource usage
- **Better Extrapolation** - RoPE handles sequences longer than training length

## 💡 Next Steps / TODOs

- Implement Byte Pair Encoding (BPE) for subword tokenization
- Add multi-GPU support
- Implement KV cache for faster generation
- Add more sophisticated sampling strategies

## 🤝 Contributing

Contributions are welcome! Feel free to submit PRs or open issues for:
- Bug fixes
- Feature additions
- Documentation improvements
- Educational clarifications

## 📝 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 👨‍💻 Author

[Shailesh2302](https://github.com/Shailesh2302)

## ⭐ Support

If you find this project helpful for understanding modern LLM architectures, please consider giving it a star!

## 📚 References

- Llama 2: [arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)
- RoPE: [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- GQA: [arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)
- SwiGLU: [arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)
