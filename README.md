# Reproducing Continuous Batching: A Token-Level Scheduling Study for LLM Inference

This repository contains our CS690 Systems for Deep Learning final project at UMass Amherst.

We reproduce the core qualitative claim of **Orca (OSDI 2022)**: continuous batching can improve large language model inference throughput over static batching under concurrent workloads.

## Project Summary

Large language models generate text autoregressively, one token at a time. Static batching keeps a fixed batch until all requests finish, which can waste GPU work. Continuous batching updates the active batch at every decoding step by removing completed requests.

In this project, we implement and compare:

- **Static batching** using HuggingFace `model.generate()`
- **Continuous batching** using a custom Python-level token scheduler
- Throughput, latency, and GPU memory behavior
- Prompt-length sensitivity for short, medium, and long prompts

## Key Result

Using **GPT-2 Medium (355M parameters)** on a **Google Colab NVIDIA T4 GPU**, continuous batching achieved up to **9% higher throughput** in our small-scale reproduction.

The gains are strongest when batch size is large enough to amortize scheduling overhead. We also observe that longer prompts reduce the relative advantage because KV-cache pruning becomes more expensive.

```markdown
## Repository Contents

```text
.
├── 690mfinal_project.pdf        # Final project report
├── 690AB_Project.ipynb          # Notebook with implementation and experiments
├── README.md
└── requirements.txt

## Notes on Reproducibility

The experiments were run on Google Colab using an NVIDIA T4 GPU. Absolute throughput values may vary across runs because Colab GPUs are shared and GPU clock states can change between sessions.

The final report records the benchmark values used for the submitted figures and tables.

## Authors

- Ankit Panda
- Hrudayaditya Jallu

University of Massachusetts Amherst  
CS690AB: Systems for Deep Learning
