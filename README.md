# Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2606.15880"><img src="https://img.shields.io/badge/ArXiv-B31B1B?logo=arxiv&logoColor=white" alt="ArXiv"></a>
  <a href="https://github.com/KQL11/Deep-VRM"><img src="https://img.shields.io/badge/Code-7F52FF?logo=github&logoColor=white" alt="Code"></a>
  <a href="https://huggingface.co/Kaiqing/Deep-VRM-Qwen-25-VL-7B/tree/main"><img src="https://img.shields.io/badge/Model-369c2b?logo=huggingface" alt="Model"></a>
  <a href="https://huggingface.co/datasets/Kaiqing/DeepVRM-Data/tree/main"><img src="https://img.shields.io/badge/Dataset-007ACC?logo=huggingface" alt="Dataset"></a>
</p>

📄 This is the official code of the paper (ICML 2026) **["Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models"](https://arxiv.org/abs/2606.15880)**.

## 📌 Release Status

This repository is a preview release. The full project release is in preparation.

- 🧪 Code: [preview version](https://github.com/KQL11/Deep-VRM)
- 📦 Training data: [DeepVRM-Data](https://huggingface.co/datasets/Kaiqing/DeepVRM-Data/tree/main)
- 🧠 Model checkpoints: [Deep-VRM-Qwen-25-VL-7B](https://huggingface.co/Kaiqing/Deep-VRM-Qwen-25-VL-7B/tree/main)
  ❗ Since the model checkpoint was trained on AMD GPUs, we are not sure whether there may be performance differences during reproduction. Please contact us if you have any questions.
- **⚠️ Warning! We fixed a bug where the Low-Level Visual Encoder was not initialized from the pretrained main visual encoder before Stage 2 training (see `Models/DeepVRM/custom_optim.py` for the fix). Please use the latest version of our code.**

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
eval.sh              Evaluation script
DATA/Test/eval_data.json  Evaluation data template used by eval.sh
```

Large training artifacts are intentionally not included in this release. The `DATA/` and `Checkpoints*/` directories are ignored by Git, except for the small evaluation template at `DATA/Test/eval_data.json`.

## ⚙️ Environment Setup

Install the local `ms-swift` package before training:

```bash
cd ms-swift
pip install -e .
```

Additional runtime dependencies include `transformers>=4.50`, `qwen_vl_utils`, `decord`, and the dependencies required by SWIFT.

## 📦 Dataset Preparation

Install the Hugging Face CLI, then run the download command from the repository root. The remote directory structure is preserved, so the images are placed directly in `DATA/Train/Prompts/DeepVRM_Image`:

```bash
pip install -U huggingface_hub
hf download Kaiqing/DeepVRM-Data \
    --repo-type dataset \
    --include "DATA/Train/Prompts/DeepVRM_Image/**" \
    --local-dir .
```

The image data is approximately 37 GB, so make sure there is enough free space. Together with the training annotations provided in this repository, the directory structure should be:

```text
DATA/
├── Train/
│   └── Prompts/
│       ├── DeepVRM_Image/          # Downloaded training images
│       │   ├── sd14/
│       │   ├── dda_coco_sd2/
│       │   └── ...
│       ├── GenImageTrain.jsonl
│       ├── multiround_semantic.jsonl
│       ├── Non-semantic.jsonl
│       ├── unrealistic.json
│       ├── dda_1.json
│       └── dda_2.json
```

Image paths in each JSON or JSONL annotation file should be relative to the repository root, for example:

```json
{
  "images": ["DATA/Train/Prompts/DeepVRM_Image/dda_coco_sd2/real/000000120400.png"],
  "messages": [
    {"role": "user", "content": "Is this a real image? <image>"},
    {"role": "assistant", "content": "yes"}
  ]
}
```

The annotation groups currently used by the two training scripts are:

```bash
# run_Stage1.sh
DATASETS="DATA/Train/Prompts/multiround_semantic.jsonl DATA/Train/Prompts/GenImageTrain.jsonl DATA/Train/Prompts/unrealistic.json"
```

```bash
# run_Stage2.sh
DATASETS="DATA/Train/Prompts/dda_1.json DATA/Train/Prompts/dda_2.json DATA/Train/Prompts/GenImageTrain.jsonl DATA/Train/Prompts/multiround_semantic.jsonl DATA/Train/Prompts/Non-semantic.jsonl DATA/Train/Prompts/unrealistic.json"
```

Keep these relative paths unchanged, or update the corresponding `DATASETS` line if the annotation files are stored elsewhere. Multiple annotation files are passed as a space-separated list.

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

## 🧪 Evaluation

Prepare one or more evaluation JSON files following the format in `DATA/Test/eval_data.json`, then place them in `DATA/Test` or pass another directory to `eval.sh`.

```bash
MODEL_PATH=Kaiqing/Deep-VRM-Qwen-25-VL-7B ./eval.sh DATA/Test
```

Results are saved to `output/` by default.

## 🙏 Acknowledgements

This project uses and builds on [modelscope/ms-swift](https://github.com/modelscope/ms-swift/) for model training and customization support. We thank the SWIFT team for their open-source work.

## 📝 Citation

Coming soon.

## 📜 License

Coming soon.
