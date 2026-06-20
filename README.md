# Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2606.15880"><img src="https://img.shields.io/badge/ArXiv-B31B1B?logo=arxiv&logoColor=white" alt="ArXiv"></a>
  <a href="https://github.com/KQL11/Deep-VRM"><img src="https://img.shields.io/badge/Code-7F52FF?logo=github&logoColor=white" alt="Code"></a>
  <a href="https://huggingface.co/Kaiqing/Deep-VRM-Qwen-25-VL-7B/tree/main"><img src="https://img.shields.io/badge/Model-369c2b?logo=huggingface" alt="Model"></a>
</p>

📄 This is the official code of the paper (ICML 2026) **["Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models"](https://arxiv.org/abs/2606.15880)**.

## 📌 Release Status

This repository is a preview release. The full project release is in preparation.

- 🧪 Code: [preview version](https://github.com/KQL11/Deep-VRM)
- 📦 Training data: coming soon
- 🧠 Model checkpoints: [Deep-VRM-Qwen-25-VL-7B](https://huggingface.co/Kaiqing/Deep-VRM-Qwen-25-VL-7B/tree/main)
  Since the model checkpoint was trained on AMD GPUs, we are not sure whether there may be performance differences during reproduction. Please contact us if you have any questions.

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
