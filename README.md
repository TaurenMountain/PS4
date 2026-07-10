# PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction

<p align="center">
  <a href="https://arxiv.org/abs/2607.08111"><img src="https://img.shields.io/badge/arXiv-2607.08111-b31b1b.svg" /></a>
  <a href="https://huggingface.co/TaurenMountain/PS4"><img src="https://img.shields.io/badge/🤗 PS4-Model-yellow" /></a>
  <a href="https://huggingface.co/datasets/TaurenMountain/REAL-PS4"><img src="https://img.shields.io/badge/🤗 REAL--PS4-Dataset-orange" /></a>
  <a href="#"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-green.svg" /></a>
  <a href="https://www.yijiahe.com/"><img src="https://img.shields.io/badge/Released%20by-Yijiahe-red" /></a>
</p>

**PS4** is a proxy-supervised training framework for target speaker extraction (TSE) in real conversational mixtures. 
To support training, we construct **[REAL-PS4](https://huggingface.co/datasets/TaurenMountain/REAL-PS4)**, a large-scale corpus of 71,771 samples derived from four public conversational datasets (AISHELL-4, AliMeeting, AMI, CHiME-6), covering both Chinese and English scenarios. On the [REAL-T Challenge](https://real-tse.github.io/challenge/) leaderboard, PS4 ranks **2nd overall**, achieving the **best speaker similarity and timing F1** among all submitted systems.

- **Model weights:** [TaurenMountain/PS4](https://huggingface.co/TaurenMountain/PS4)
- **Training dataset:** [TaurenMountain/REAL-PS4](https://huggingface.co/datasets/TaurenMountain/REAL-PS4)

## 🔥 News

- **2026-07**: PS4 ranks **2nd overall** on the [REAL-T Challenge](https://real-tse.github.io/challenge/) leaderboard, achieving the **best speaker similarity (SIM) and timing F1** among all submitted systems!
- **2026-07**: Paper [*PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction*](https://arxiv.org/abs/2607.08111) uploaded to arXiv.
- **2026-07**: Training code and [REAL-PS4](https://huggingface.co/datasets/TaurenMountain/REAL-PS4) dataset released.

## 🏆 REAL-T Challenge Leaderboard

Results on the official REAL-T challenge validation set. The ranking metric is a composite score across TER, F1, SIM, and DNSMOS-P808.

| Rank | System | TER ↓ | F1 ↑ | SIM ↑ | DNSMOS-P808 ↑ |
|------|--------|--------|-------|-------|----------------|
| 🥇 1st | MERL's | **0.613** | 0.861 | 0.538 | **3.371** |
| 🥈 2nd | **PS4 (ours)** | 0.639 | **0.871** | **0.565** | 3.128 |
| 🥉 3rd | CARTSE's | 0.651 | 0.857 | 0.544 | 3.138 |
| — | BSRNN\_EMB (baseline) | 0.829 | 0.829 | 0.417 | 2.875 |
| — | BSRNN\_TFMAP (baseline) | 0.838 | 0.829 | 0.443 | 2.756 |

> PS4 achieves the **best F1 (0.871)** and **best SIM (0.565)** among all submitted systems.

## Repository Structure

```
.
├── train.py                          # Main training script (single-GPU & multi-GPU DDP)
├── resume_utils.py                   # Checkpoint resume utilities
├── run_train.sh                      # Training launcher (handles single/multi-GPU, resume/finetune)
└── configs/
    ├── config_bsrnn_ecapa_vox1.yaml  # PS4 training config (BSRNN + ECAPA-TDNN)
    └── config_tfmap_context_100.yaml # Alternative TF-Map model config
```

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.0
- torchaudio
- transformers (Whisper large-v3)
- wesep (from [REAL-TSE-Challenge](https://github.com/wenet-e2e/wesep))
- pandas, numpy, pyyaml, tqdm, tensorboard

Install:
```bash
pip install torch torchaudio transformers pandas numpy pyyaml tqdm tensorboard
```

The BSRNN model implementation is sourced from [wesep](https://github.com/wenet-e2e/wesep). Clone it alongside this repo:
```bash
git clone https://github.com/wenet-e2e/wesep REAL-TSE-Challenge/wesep_real_tse
```

## Quick Start

### 1. Prepare Data

Download [TaurenMountain/REAL-PS4](https://huggingface.co/datasets/TaurenMountain/REAL-PS4) and set `data.train_roots` in the config to point to your local copy.

### 2. Edit Config

Edit [`configs/config_bsrnn_ecapa_vox1.yaml`](configs/config_bsrnn_ecapa_vox1.yaml) to set:

```yaml
pretrained_tse: /path/to/bsrnn_ecapa_vox1/avg_model.pt   # pretrained TSE backbone
whisper_model_path: /path/to/whisper-large-v3             # Whisper ASR model
spk_encoder_en_path:  /path/to/voxceleb_resnet34_LM       # EN speaker encoder
spk_encoder_chs_path: /path/to/cnceleb_resnet34_LM        # ZH speaker encoder
dnsmos_model_dir: /path/to/DNSMOS                         # DNSMOS ONNX models

data:
  train_roots:
    - /path/to/REAL-PS4
```

### 3. Train

**Single GPU:**
```bash
bash run_train.sh --model bsrnn_ecapa_vox1 --gpus 0
```

**Multi-GPU (DDP):**
```bash
bash run_train.sh --model bsrnn_ecapa_vox1 --gpus 0,1,2,3
```

**Resume from latest checkpoint:**
```bash
bash run_train.sh --model bsrnn_ecapa_vox1 --resume --gpus 0,1,2,3
```

**Fine-tune from an existing experiment:**
```bash
bash run_train.sh --model exp/20260619_174045_bsrnn_ecapa_vox1 --gpus 0,1
```

Experiment outputs are saved to `exp/<timestamp>_<model>/`:
```
exp/<timestamp>_<model>/
├── config.yaml        # copy of config used
├── train.log          # training log
├── models/            # checkpoints (checkpoint_epochXXX.pt)
└── tensorboard/       # TensorBoard events
```

## Training Objective

PS4 uses a **combined proxy-supervised loss**:

```
L = λ_ce · L_CE  +  λ_sim · L_sim  +  λ_vad · L_VAD  +  λ_dnsmos · L_DNSMOS
```

| Loss | Default weight | Description |
|------|---------------|-------------|
| `L_CE` | 1.0 | Whisper large-v3 ASR cross-entropy (teacher-forcing) |
| `L_sim` | 5.0 | Speaker similarity ranking loss (hinge, margin=0.5) |
| `L_VAD` | 0.5 | Target speaker activity detection (frame-level energy) |
| `L_DNSMOS` | 0.2 | Differentiable DNSMOS-OVRL (no reference audio needed) |

Set `loss_mode: ce | similarity | combined` in the config to select which losses to use.

## Citation

```bibtex
@misc{ning2026ps4,
      title={PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction}, 
      author={Wanyi Ning and Wei Zhou and Yingpeng Li and Yinshang Guo and Haitao Qian and Yiming Cheng},
      year={2026},
      eprint={2607.08111},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2607.08111}, 
}
```
