# Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models

📄 This is the official code of the paper (ICML 2026) **"Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models"**.

## 📌 Release Status

This repository is a preview release. The full project release is in preparation.

- 🧪 Code: preview version
- 📦 Training data: coming soon
- 🧠 Model checkpoints: coming soon

## 🔍 Overview

This project builds on Qwen2.5-VL and introduces a DeepVRM-style residual injection design for multimodal forensic signal perception. The training pipeline is organized into two stages:

- Stage 1 trains on the original Qwen2.5-VL model.
- Stage 2 trains the customized DeepVRM model with residual low-level visual features.

## 📁 Repository Structure

```text
Models/DeepVRM/      DeepVRM model, processor, registration, and custom tuner code
ms-swift/            Local SWIFT training framework used by the scripts
run_Stage1.sh        Stage 1 training script
run_Stage2.sh        Stage 2 training script
```

Large training artifacts are intentionally not included in this release. The `DATA/` and `Checkpoints*/` directories are ignored by Git and will be released separately.

## ⚙️ Environment Setup

Install the local `ms-swift` package before training:

```bash
cd ms-swift
pip install -e .
```

Additional runtime dependencies include `transformers>=4.50`, `qwen_vl_utils`, `decord`, and the dependencies required by SWIFT.

## 🚀 Training

Stage 1:

```bash
bash run_Stage1.sh
```

Stage 2:

```bash
bash run_Stage2.sh
```

Before running the scripts, update the dataset paths, checkpoint paths, GPU settings, and batch sizes according to your environment.

## 🙏 Acknowledgements

This project uses and builds on [modelscope/ms-swift](https://github.com/modelscope/ms-swift/) for model training and customization support. We thank the SWIFT team for their open-source work.

## 📝 Citation

Coming soon.

## 📜 License

Coming soon.
