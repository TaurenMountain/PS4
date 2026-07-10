#!/usr/bin/env bash
# ============================================================
# TSE-ASR 端到端训练启动脚本（支持单卡 & 多卡 DDP）
#
# 用法:
#   # 从预训练模型名开始全新训练
#   bash run_train.sh --model bsrnn_ecapa_vox1 --gpus 0,1,4,5
#   bash run_train.sh --model tfmap_context_100 --gpus 0
#
#   # 从已有 exp 目录加载配置+权重，创建新实验（finetune）
#   bash run_train.sh --model exp/20240101_120000_bsrnn_ecapa_vox1 --gpus 0,1
#
#   # 继续在原 exp 目录训练（resume）
#   bash run_train.sh --model exp/20240101_120000_bsrnn_ecapa_vox1 --resume --gpus 0,1,4,5
#
#   # 模型名 + resume：在最近一次的对应实验目录继续训练
#   bash run_train.sh --model bsrnn_ecapa_vox1 --resume --gpus 0,1,4,5
#
# 参数说明:
#   --model   <name|path>  模型名（bsrnn_ecapa_vox1 / tfmap_context_100）
#                          或 exp/ 下已有的实验目录路径
#   --resume               开关，使用时在 --model 对应目录原地继续训练
#   --gpus    <ids>        逗号分隔的 GPU 编号，例如 0,1,4,5
# ============================================================

set -euo pipefail

# ── 脚本目录 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 已知预训练模型名 → 目录映射 ──────────────────────────────────────────
PRETRAINED_ROOT="/home/yuque3/nwy/real-t/REAL-TSE-Challenge/pretrained"
EXP_ROOT="$SCRIPT_DIR/exp"
KNOWN_MODELS=("bsrnn_ecapa_vox1" "tfmap_context_100")

# ── 参数解析（getopt 风格手动解析）────────────────────────────────────────
MODEL=""
RESUME=0
GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] 未知参数: $1"
            echo "用法: bash run_train.sh --model <name|path> [--resume] [--gpus 0,1,4,5]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "[ERROR] 必须指定 --model 参数（模型名或 exp 目录路径）"
    echo "用法: bash run_train.sh --model <name|path> [--resume] [--gpus 0,1,4,5]"
    exit 1
fi

# ── 辅助函数：找 models/ 下最新的 .pt checkpoint ─────────────────────────
find_latest_ckpt() {
    local models_dir="$1"
    # 优先按文件修改时间最新的 .pt
    ls -t "$models_dir"/*.pt 2>/dev/null | head -1 || true
}

# ── 判断 MODEL 是名字还是路径 ─────────────────────────────────────────────
IS_NAME=0
for n in "${KNOWN_MODELS[@]}"; do
    if [[ "$MODEL" == "$n" ]]; then
        IS_NAME=1
        break
    fi
done

CONFIG_FILE=""
RESUME_CKPT=""
EXP_DIR_OVERRIDE=""

if [[ "$IS_NAME" -eq 1 ]]; then
    # ── 模型名模式 ──────────────────────────────────────────────────────
    CONFIG_FILE="$SCRIPT_DIR/configs/config_${MODEL}.yaml"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "[ERROR] 配置文件不存在: $CONFIG_FILE"
        echo "可用配置："
        ls "$SCRIPT_DIR/configs/"*.yaml 2>/dev/null | xargs -I{} basename {} .yaml || echo "  (无)"
        exit 1
    fi

    if [[ "$RESUME" -eq 1 ]]; then
        # 找 exp/ 下最近一次以 _<MODEL> 结尾的实验目录
        LATEST_EXP=$(ls -dt "$EXP_ROOT"/*_"${MODEL}" 2>/dev/null | head -1 || true)
        if [[ -z "$LATEST_EXP" ]]; then
            echo "[ERROR] --resume 模式：未在 $EXP_ROOT 下找到匹配 *_${MODEL} 的实验目录"
            exit 1
        fi
        LATEST_CKPT=$(find_latest_ckpt "$LATEST_EXP/models")
        if [[ -z "$LATEST_CKPT" ]]; then
            echo "[ERROR] 实验目录 $LATEST_EXP/models 下未找到 .pt checkpoint"
            exit 1
        fi
        EXP_DIR_OVERRIDE="$LATEST_EXP"
        RESUME_CKPT="$LATEST_CKPT"
        echo "[INFO] resume 模式：继续实验目录 $LATEST_EXP"
        echo "[INFO] 加载 checkpoint: $LATEST_CKPT"
    fi

else
    # ── 路径模式 ────────────────────────────────────────────────────────
    # 支持相对路径（相对于 SCRIPT_DIR）
    if [[ ! -d "$MODEL" ]]; then
        ABS_MODEL="$SCRIPT_DIR/$MODEL"
        if [[ -d "$ABS_MODEL" ]]; then
            MODEL="$ABS_MODEL"
        else
            echo "[ERROR] 找不到实验目录: $MODEL"
            exit 1
        fi
    fi
    MODEL="$(cd "$MODEL" && pwd)"  # 转绝对路径

    CONFIG_FILE="$MODEL/config.yaml"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "[ERROR] 实验目录下未找到 config.yaml: $CONFIG_FILE"
        exit 1
    fi

    LATEST_CKPT=$(find_latest_ckpt "$MODEL/models")
    if [[ -z "$LATEST_CKPT" ]]; then
        echo "[ERROR] $MODEL/models 下未找到 .pt checkpoint"
        exit 1
    fi
    RESUME_CKPT="$LATEST_CKPT"

    if [[ "$RESUME" -eq 1 ]]; then
        # 原地继续训练：传入 exp_dir 防止生成新时间戳目录
        EXP_DIR_OVERRIDE="$MODEL"
        echo "[INFO] resume 模式：原地继续实验目录 $MODEL"
    else
        echo "[INFO] finetune 模式：从 $LATEST_CKPT 初始化，创建新实验目录"
    fi
    echo "[INFO] 加载 checkpoint: $LATEST_CKPT"
fi

# ── GPU 设置 ──────────────────────────────────────────────────────────────
if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    # 计算 GPU 数量（逗号分隔的个数）
    NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
else
    # 自动检测
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        NUM_GPUS=$((NUM_GPUS > 0 ? NUM_GPUS : 1))
    else
        NUM_GPUS=1
    fi
fi

# ── Python ────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "[ERROR] Python 未找到: $PYTHON"
    exit 1
fi

# ── 环境变量：将 wesep 加入 PYTHONPATH ────────────────────────────────────
WESEP_PATH="$SCRIPT_DIR/../REAL-TSE-Challenge/wesep_real_tse"
export PYTHONPATH="$WESEP_PATH:${PYTHONPATH:-}"

# ── 打印信息 ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  TSE-ASR 端到端训练"
echo "  配置文件  : $CONFIG_FILE"
echo "  GPU 编号  : ${GPUS:-auto}"
echo "  GPU 数量  : $NUM_GPUS"
[[ -n "$EXP_DIR_OVERRIDE" ]] && echo "  实验目录  : $EXP_DIR_OVERRIDE"
[[ -n "$RESUME_CKPT"      ]] && echo "  Resume    : $RESUME_CKPT"
echo "  启动时间  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── 构建 train.py 参数 ────────────────────────────────────────────────────
TRAIN_ARGS=(--config "$CONFIG_FILE")
[[ -n "$EXP_DIR_OVERRIDE" ]] && TRAIN_ARGS+=(--exp_dir "$EXP_DIR_OVERRIDE")
[[ -n "$RESUME_CKPT"      ]] && TRAIN_ARGS+=(--resume  "$RESUME_CKPT")

# ── 启动训练 ─────────────────────────────────────────────────────────────
if [[ "$NUM_GPUS" -le 1 ]]; then
    echo "[INFO] 单卡模式: $PYTHON train.py ${TRAIN_ARGS[*]}"
    exec "$PYTHON" train.py "${TRAIN_ARGS[@]}"
else
    TORCHRUN="${TORCHRUN:-torchrun}"
    if ! command -v "$TORCHRUN" &>/dev/null; then
        echo "[WARN] torchrun 未找到，尝试 python -m torch.distributed.run"
        TORCHRUN="$PYTHON -m torch.distributed.run"
    fi

    MASTER_ADDR="${MASTER_ADDR:-localhost}"
    MASTER_PORT="${MASTER_PORT:-29500}"

    echo "[INFO] 多卡 DDP 模式: ${TORCHRUN} --nproc_per_node=${NUM_GPUS} train.py ${TRAIN_ARGS[*]}"
    exec $TORCHRUN \
        --nproc_per_node="$NUM_GPUS" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        train.py "${TRAIN_ARGS[@]}"
fi
