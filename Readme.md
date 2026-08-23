# BitLLM: Design and Analysis of 1-Bit and 1.58-Bit Large Language Models

Exploring **1-bit and 1.58-bit neural networks** and their application to Transformer-based Large Language Models, with a focus on reducing memory footprint and computational requirements while preserving useful model performance.

> **Project:** BitLLM
> **Institution:** Faculty of Technology, University of Delhi
> **Department:** Electronics & Communication Engineering
> **Academic Year:** 2025–2026

---

## Overview

Large Language Models rely heavily on high-precision numerical computation, resulting in substantial memory, computational, and energy requirements. **BitLLM** investigates low-precision neural network architectures as an alternative, focusing on **Binary Neural Networks (BNNs)** and **Ternary Neural Networks (TNNs)** with weights constrained to:

* **1-bit:** `{-1, +1}`
* **1.58-bit:** `{-1, 0, +1}`

The project progresses from low-bit feed-forward neural networks evaluated on **MNIST** to a **GPT-2-style Transformer** incorporating low-bit computation.

The work investigates the trade-off between model accuracy, training stability, memory efficiency, and computational requirements when reducing numerical precision.

---

## Key Objectives

* Implement Binary Neural Networks with 1-bit weights.
* Implement Ternary Neural Networks with 1.58-bit weights.
* Compare low-bit models against full-precision baselines.
* Analyze the effect of network width and depth on low-bit model performance.
* Extend low-bit techniques to a GPT-style Transformer architecture.
* Investigate quantized weights and activations for efficient computation.
* Evaluate the trade-off between predictive performance and resource efficiency.
* Explore techniques for improving the stability of low-bit Transformer training.

---

## Architecture

The project consists of two major experimental stages.

### 1. Binary and Ternary Neural Networks

Low-bit networks were first implemented and evaluated on MNIST using multiple architectures:

```text
784 → 128 → 64 → 10

784 → 256 → 128 → 10

784 → 256 → 128 → 64 → 10
```

Weights are quantized using binary and ternary representations, while the **Straight-Through Estimator (STE)** is used to approximate gradients during backpropagation.

### 2. GPT-Style Low-Bit Transformer

The project was subsequently extended to a GPT-2-style Transformer containing:

* Token embeddings
* Positional encoding
* Multi-head self-attention
* Low-bit feed-forward layers
* Output projection
* Sub-Layer Normalization (SubLN)
* Squared ReLU activation
* 8-bit activation quantization

The Transformer architecture and its low-bit modifications are illustrated in the project report.

---

## Quantization

### Binary Quantization

Binary weights are constrained to:

```text
{-1, +1}
```

This allows conventional multiplication operations to be represented through simpler addition and subtraction operations.

### Ternary Quantization

Ternary weights are constrained to:

```text
{-1, 0, +1}
```

The additional zero state provides greater representational flexibility and can introduce sparsity by eliminating less important connections.

### Straight-Through Estimator

Since quantization is non-differentiable, the project uses the **Straight-Through Estimator (STE)** to approximate gradients during training.

---

## Experimental Results

### MNIST

The initial experiments compare full-precision, 1-bit, and 1.58-bit networks.

| Metric              | Full Precision |         1-bit | 1.58-bit |
| ------------------- | -------------: | ------------: | -------: |
| Test Accuracy       |      **97.1%** |         91.0% |    91.9% |
| Validation Accuracy |      **97.5%** |         91.9% |    92.7% |
| Training Loss       |      **0.053** |         1.332 |    1.249 |
| Validation Loss     |      **0.078** |         1.291 |    1.129 |
| Model Efficiency    |            Low | **Very High** |     High |

The experiments demonstrate that low-bit models trade some predictive performance for substantially improved model efficiency.

### Effect of Network Width

For the wider architecture:

```text
784 → 256 → 128 → 10
```

the 1.58-bit model achieved **94.6% test accuracy**, compared with **93.3%** for the 1-bit model.

### Effect of Network Depth

For the deeper architecture:

```text
784 → 256 → 128 → 64 → 10
```

the 1-bit model achieved **94.8% test accuracy**, while the 1.58-bit model achieved **93.2%**. This demonstrates that the relative behavior of binary and ternary networks depends on architectural depth and width.

---

## BitNet 1.58 Transformer Results

A GPT-style BitNet 1.58 model was compared against a full-precision baseline with the same total number of parameters.

| Metric                      | BitNet 1.58 | FP32 Baseline |
| --------------------------- | ----------: | ------------: |
| Best Validation Loss        |      4.5270 |    **4.4143** |
| Final Validation Loss       |      4.5270 |    **4.4145** |
| Final Validation Accuracy   |      29.99% |    **31.55%** |
| Final Validation Perplexity |       92.48 |     **82.64** |
| Training Time               |      1883 s |    **1770 s** |
| Parameters                  |  16,275,968 |    16,275,968 |
| BitNet Memory Ratio vs FP32 |  **80.10%** |          100% |

The BitNet model exhibits a modest performance degradation relative to the FP32 baseline while reducing its effective memory footprint. The experiments therefore demonstrate a **performance-efficiency trade-off**, rather than claiming that low-bit quantization universally improves model performance.

---

## Low-Bit Transformer Design

The BitNet-style implementation incorporates several techniques intended to improve training stability and efficiency:

```text
Transformer
│
├── Token Embedding
├── Positional Representation
│
├── Transformer Blocks
│   ├── Multi-Head Self-Attention
│   ├── Low-Bit Linear Layers
│   ├── Sub-Layer Normalization
│   ├── Low-Bit Feed-Forward Network
│   ├── Squared ReLU
│   └── 8-bit Activation Quantization
│
└── Language Modeling Head
```

The appendix additionally documents an implementation using ternary weight quantization, per-token 8-bit activation quantization, STE, SwiGLU, RoPE, RMSNorm, and full-precision embeddings and language-modeling head.

---

## Project Structure

```text
BitLLM/
│
├── README.md
├── notebook/
│   └── BitLLM_Training.ipynb
│
├── project_report/
│   └── BitLLM_Project_Report.pdf
│
├── requirements.txt
└── LICENSE
```

### Notebook

`BitLLM_Training.ipynb` contains the implementation, training experiments, evaluation, and visualizations.

### Project Report

The `project_report/` directory contains the complete academic report documenting the methodology, architecture, experiments, results, analysis, references, and future scope.

---

## Tech Stack

* **Python**
* **PyTorch**
* **TensorFlow**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Jupyter Notebook**
* **CUDA** for GPU acceleration when available

The project used Python and modern machine learning frameworks for implementing custom quantization operations and training the neural networks.

---

## Reproducibility

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open the training notebook:

```text
notebook/BitLLM_Training.ipynb
```

The experiments may produce slightly different results depending on hardware, initialization, and training conditions.

---

## Research Direction

BitLLM serves as an exploration of **extreme low-precision deep learning** and the potential of 1-bit and 1.58-bit models for resource-constrained AI systems.

Future directions include:

* Scaling to larger language models
* Hardware accelerators for low-bit operations
* Improved quantization algorithms
* More stable training methods
* Edge and embedded deployment
* Applications beyond language modeling
* Hardware-aware neural network design

These directions are particularly relevant to the development of efficient AI systems where memory, computation, and energy are constrained.

---

## References

The project is primarily based on research surrounding BitNet, binary neural networks, ternary neural networks, and Transformer architectures.

1. Wang et al., **"BitNet: Scaling 1-bit Transformers for Large Language Models,"** 2023.
2. Ma et al., **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits,"** 2024.
3. Radford et al., **"Language Models are Unsupervised Multitask Learners,"** 2019.
4. Vaswani et al., **"Attention Is All You Need,"** NeurIPS, 2017.
5. Courbariaux et al., **"BinaryConnect: Training Deep Neural Networks with Binary Weights during Propagations,"** NeurIPS, 2015.

The complete reference list is available in the project report.

---

## Authors

**Manish Sharma**
**Aayush Chauhan**
**Ayush Kumar**
**Pragyansh Verma**

Department of Electronics & Communication Engineering
Faculty of Technology, University of Delhi

### Supervisors

**Prof. Vanita Jain**
Professor

**Dr. Sangeeta Yadav**
Assistant Professor

---

## License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for the full license text.

