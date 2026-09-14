# LogosTransformer

A decoder-only (GPT-style) transformer implemented from scratch in PyTorch and trained at the character level on a corpus of philosophical texts.

## Motivation

The goal was to understand transformer architectures by building one, rather than by calling one. No HuggingFace, no `nn.Transformer`, no pretrained weights. Attention, masking, normalization, residual paths, positional embeddings, the training loop, and the sampling logic are all written directly.

Character-level tokenization was chosen deliberately. It removes subword tokenization as a source of "hidden" structure, which makes the emergence of coherent language attributable to the model rather than to the tokenizer.

## What is implemented

| Component | Notes |
|---|---|
| Character-level tokenizer | Vocabulary built from unique characters in the corpus |
| Scaled dot-product attention | Implemented manually, including the `1/sqrt(d_k)` scaling |
| Causal masking | Lower-triangular mask preventing attention to future positions |
| Multi-head attention | Parallel heads, concatenated and projected |
| Position-wise feed-forward network | Two linear layers with a 4x inner expansion |
| Layer normalization and residual connections | Applied around both sub-layers of each block |
| Learned positional embeddings | Added to token embeddings at the input |
| Training loop | AdamW optimizer, learning rate schedule, periodic loss estimation over held-out batches |
| Checkpointing | Save and resume model state |
| Text generation | Autoregressive sampling with temperature control |

## Architecture

```
input token ids
  -> token embedding + learned positional embedding
  -> N x transformer block:
       layer norm -> multi-head causal self-attention -> residual add
       layer norm -> feed-forward network            -> residual add
  -> final layer norm
  -> linear projection to vocabulary
  -> softmax over next-character distribution
```

## Configuration

```python
block_size    = 64      # context window, in characters
batch_size    = 64
n_embd        = 384     # embedding dimension
n_head        = 8       # attention heads per block
n_layer       = 4       # transformer blocks
dropout       = 0.2
learning_rate = 3e-4
max_iters     = 2000
```

Approximately 7M trainable parameters.

## Results

Training loss falls sharply over the first few hundred steps and continues to converge smoothly through step 2000.

<p align="center">
  <img src="./assets/loss_curve.png" width="600">
</p>

The more interesting result is qualitative. Given the same prompt, output at initialization versus output at step 2000:

**Step 0**

```
Hello! Can you see me?Mf5Ejæ'Pb-S8wx 4!9DK—wiPdjfouJ,L"p7WxVx7fBHtU 7i]HRaEx,TIHJGXæ—eIk5p7—;N*"qT=expR!YWBqEvt"]UM?:]!-;IqlWr5e?RelxlgNr?f9''eseeq]I0*AClX??qWw;)7;lJ'l'Ygp':f?;JlgU=BQYZ'U?JHuZ?5RaL"FcR7Zl?3xJHæb2;rgI
```

**Step 2000**

```
Hello! Can you see me? I ought to be absolute to one, and when he
can say that I can see to be considered him, to a concerned would as
the money, then, say, although they do not allow those who examine
the same are objects
```

Nothing about word boundaries, spelling, or syntax was supplied to the model. All of it was learned character by character from the training corpus.

## Repository structure

```
├── assets/
│   └── loss_curve.png
├── data/
│   └── philosophers.txt         # training corpus
├── checkpoints/                 # saved model weights
├── gpt-mk03-updated3.ipynb      # training and generation notebook
├── requirements.txt
└── README.md
```

## Running the model

```bash
pip install -r requirements.txt
jupyter notebook gpt-mk03-updated3.ipynb
```

Training was run on a single machine without distributed setup. Reduce `max_iters` or `n_layer` for a faster pass.

## Scope and limitations

This is a learning artifact, not a production language model. Specifically:

- Character-level tokenization caps the effective information density per token compared to BPE or SentencePiece.
- A 64-character context window is short enough that the model cannot track long-range argument structure.
- At 2000 steps and ~7M parameters, the model produces locally fluent text without semantic grounding.
- Evaluation is limited to training and validation loss. There are no downstream benchmarks.

## Roadmap

1. Replace character-level tokenization with byte-pair encoding and measure the effect on validation loss at fixed compute.
2. Add an evaluation layer beyond loss, including perplexity on a held-out corpus and simple probes for syntactic consistency.
3. Extend context length and depth, and profile the memory and throughput tradeoff.
4. Add a parameter-efficient fine-tuning path (LoRA) on top of the trained base.
5. Serve the model behind a small inference endpoint for interactive generation.

## License

MIT

## Author

Carlos Santa. Designed and implemented independently.
