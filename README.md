# 📘 **LogosTransformer — A GPT-Style Language Model Trained on Philosophy Texts**

## 🧠 Overview
**LogosTransformer** is a miniature GPT-style transformer model trained **from scratch** in PyTorch using a **character-level tokenizer** and a corpus of philosophical writings (Aristotle, Socrates, Kant, Descartes, etc.).

This project is built entirely inside a Jupyter Notebook for transparency and educational value — no HuggingFace, no shortcuts.  
By implementing attention, feed-forward layers, positional embeddings, and sampling logic from the ground up, this project demonstrates a working understanding of how GPT-style large language models operate internally.

The most compelling outcome is how **coherent, philosophy-inspired language emerges** from purely **character-level** training.

---

# 🚀 Features

### 🔡 Character-Level Tokenization  
- Vocabulary consists of unique characters from the corpus  
- Demonstrates how structure and meaning arise from minimal tokenization  
- Enables training on arbitrarily large text files

### 🔥 Full Transformer Implementation (No External Libraries)
Implemented manually:
- Scaled dot-product attention  
- Multi-head attention  
- Feed-forward layers  
- Layer normalization  
- Residual connections  
- Learned positional embeddings  
- Training loop w/ optimizer, scheduler, and loss estimation  
- Checkpoint saving + loading  

### ✍️ Text Generation
- Temperature-controlled sampling  
- Produces increasingly coherent output as training progresses  
- Captures philosophical tone despite char-level encoding

---

# 📉 Training Progress

The model shows strong learning behaviour, rapidly improving in the first few hundred steps and continuing to converge smoothly.

<p align="center">
  <img src="./assets/loss_curve.png" width="600">
</p>

---

# ✍️ Emergence of Coherent Philosophical Language

### **🧪 Output at Start of Training**

```
Hello! Can you see me?Mf5Ejæ’Pb-S8wx 4!9DK—wiPdjfouJ,L"p7WxVx7fBHtU 7i]HRaEx,﻿TIHJGXæ—eI﻿k5p7—;N*”qT=expR!YWBqEvt"]UM?:]﻿!-;IqlWr5e?RelxlgNr?f9'’eseeq]I0*AClX??qWw;)7;lJ’l'Ygp':f?;JlgU=BQYZ'U?JHuZ?5RaL“Fc﻿R7Zl?3xJHæb2;rgI
```

---

### **🧪 Output at Step 2000**

```
Hello! Can you see me? I ought to be absolute to one, and when he
can say that I can see to be considered him, to a concerned would as
the money, then, say, although they do not allow those who examine
the same are objects
```

---

# 🧩 Architecture Summary

```
Token Embeddings  
+ Positional Embeddings
→ Multi-Head Self-Attention
→ Feedforward Network
→ LayerNorm + Residuals
→ Linear Projection
→ Softmax over next-token distribution
```

---

# 📁 Repository Structure

```
├── assets/
│   └── loss_curve.png
├── data/
│   └── philosophers.txt
├── gpt-mk03-updated3.ipynb        # Main training notebook
├── checkpoints/                   # Optional saved weights
├── README.md
└── requirements.txt
```

---

# ⚙️ Hyperparameters Used

```python
block_size    = 64
batch_size    = 64
learning_rate = 3e-4

n_embd  = 384
n_head  = 8
n_layer = 4
dropout = 0.2

max_iters = 2000
```

---

# 🔮 Next Steps

### **1️⃣ Fine-Tuning Framework**
### **2️⃣ Upgrade Tokenization**
### **3️⃣ Expand Model Depth**
### **4️⃣ Build an Interactive Web Demo**
### **5️⃣ Add Evaluation Tools**

---

# 📜 License
MIT License

---

# 🙌 Acknowledgements
Model designed and implemented entirely by **Carlos Santa**.
