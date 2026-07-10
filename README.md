# PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction

Training code for the PS4 model — a BSRNN + ECAPA-TDNN Target Speaker Extraction (TSE) system fine-tuned with proxy supervision from Whisper large-v3 ASR on the [REAL-PS4](https://huggingface.co/datasets/TaurenMountain/REAL-PS4) dataset.

- **Model weights:** [TaurenMountain/PS4](https://huggingface.co/TaurenMountain/PS4)
- **Training dataset:** [TaurenMountain/REAL-PS4](https://huggingface.co/datasets/TaurenMountain/REAL-PS4)

## Repository Structure

```
.
├── train.py                          # Main training script (single-GPU & multi-GPU DDP)
├── resume_utils.py                   # Checkpoint resume utilities
├── run_train.sh                      # Training launcher (handles single/multi-GPU, resume/finetune)
├── configs/
│   ├── config_bsrnn_ecapa_vox1.yaml  # PS4 training config (BSRNN + ECAPA-TDNN)
│   └── config_tfmap_context_100.yaml # Alternative TF-Map model config
└── tests/
    ├── test_bsrnn_model_structure.py
    └── test_resume_utils.py
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
@article{ning2026ps4,
  title   = {PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction},
  author  = {Wanyi Ning and Wei Zhou and Yingpeng Li and Yinshang Guo and qianhaitao and Yiming Cheng},
  year    = {2026},
  publisher = {Arxiv}
}
```
