# GPT from Scratch — Bilingual Byte-Level Language Model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19oxumXz5n4_dolt28GmlrZXdfTnCYYRX?usp=sharing)

> **Note:** This is a text completion model, not a chatbot. Give it a prompt and it continues the text. It does not answer questions or follow instructions.

A decoder-only Transformer (GPT-style) built entirely from scratch in PyTorch, trained on classical Bangla and English literature. No tokenizer libraries, no pretrained weights — every component is written by hand.

---

## What This Is

This is a learning project focused on understanding and implementing the Transformer architecture from the ground up. The model reads and generates raw bytes (0–255), making it natively multilingual without any special tokenization. It was trained on ~185 MB of literary text — classical Bangla works from `barunsaha/bangla_sahitya` and English books from `lucadiliello/bookcorpusopen`.

---

## Model Architecture

| Component | Detail |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Tokenization | Byte-level (vocab size = 256) |
| Parameters | ~25.6M |
| Embedding size | 512 |
| Layers | 8 Transformer blocks |
| Attention heads | 8 |
| Block size | 256 bytes |
| Dropout | 0.1 |

---

## Training

| Detail | Value |
|---|---|
| Dataset | ~185 MB (Bangla + English, balanced) |
| Epochs | 5 |
| Batch size | 64 |
| Learning rate | 3e-4 (AdamW) |
| Hardware | Google Colab T4 GPU + Kaggle P100 |
| Final val loss | ~1.2479 |

The dataset is structured as:
```
[92 MB Bangla literary text]
================================================================================
[92 MB English literary text]
```

---

## Project Structure

```
gpt-implementation-from-scratch/
├── tiny_transformer.ipynb   # Main notebook (full pipeline)
├── combined.txt             # Not included (185 MB) — generate using the notebook
├── Final_Epoch.pt           # Not included (97 MB) — download from Releases
└── README.md
```

---

## Getting Started

### Requirements

```bash
pip install torch datasets
```

### Load the Trained Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Paste class definitions here:
# ByteTokenizer, CausalSelfAttention, FeedForward, Block, GPT
# (all found in tiny_transformer.ipynb)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("Final_Epoch.pt", map_location=device)
model = GPT(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tok = ByteTokenizer()
```

### Generate Text

```python
def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    idx = tok.encode(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    return tok.decode(out[0].cpu())[len(prompt):]

# English prompt
print(generate("The old man walked towards the river and"))

# Bangla prompt
print(generate("একদিন সন্ধ্যাবেলা নদীর ধারে"))
```

### Temperature Guide

| Temperature | Effect |
|---|---|
| 0.3 | Focused, repetitive |
| 0.7 | Balanced |
| 0.8 | Default, good variety |
| 1.2 | Creative, unpredictable |

---

## Running the CLI

A command-line text completion interface is included in the notebook. After loading the model, run the CLI cell and interact with it directly:

```
Prompt: The story begins with
Output: ...a young woman who had never seen the sea before...

Prompt: একদিন রাজার ছেলে
Output: ...বললেন, আমি এই রাজ্য ছেড়ে যাব না...
```

Commands inside the CLI:
- `:temp 0.5` — change temperature
- `:tokens 400` — change output length
- `:quit` — exit

---

## What the Model Can and Cannot Do

**Can do:**
- Continue text in both Bangla and English
- Generate grammatically structured sentences
- Maintain the literary style of the training data

**Cannot do:**
- Answer questions
- Hold a conversation
- Follow instructions

This is a base language model, not a chatbot. Making it conversational would require supervised fine-tuning on instruction/response pairs.

---

## Training on Your Own Data

1. Replace `combined.txt` with your own UTF-8 text file
2. Open `tiny_transformer.ipynb` in Google Colab
3. Mount your Google Drive and update the paths
4. Run all cells from the Dataset Preparation section onward

Checkpoints are saved to Drive after every epoch so disconnections do not wipe your progress.

---

## Key Implementation Details

**Byte-level tokenization** — no external tokenizer needed. Text is encoded directly as UTF-8 bytes, making the model handle any language or script without modification.

**Causal self-attention** — uses `F.scaled_dot_product_attention` with `is_causal=True` for efficient masked attention.

**Mixed precision training** — uses `torch.amp.autocast` and `GradScaler` for faster training on GPU.

**Gradient clipping** — clips at norm 1.0 to prevent exploding gradients.

**AdamW optimizer** — with weight decay 0.1 and betas (0.9, 0.95).

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) — GPT-2 paper
- [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT) — inspiration for the architecture
- Dataset: [barunsaha/bangla_sahitya](https://huggingface.co/datasets/barunsaha/bangla_sahitya)
- Dataset: [lucadiliello/bookcorpusopen](https://huggingface.co/datasets/lucadiliello/bookcorpusopen)

---

## Author

**Tafsir Ahmed**
[GitHub](https://github.com/tafsir99-gg)
