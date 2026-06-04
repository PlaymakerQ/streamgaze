#!/bin/bash

# activate env if needed
# source ~/miniconda3/bin/activate streamgaze

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"
export DECORD_EOF_RETRY_MAX=40960

# ===== Model Path =====
BASE_MODEL="/home/qhl/code/gaze-mllms/StreamGaze/models/ViSpeak-s3"
export VISPEAK_MODEL_PATH="$BASE_MODEL"
export VISPEAK_LORA_PATH=""

# ===== Dataset Path =====
QA_DIR="$ROOT_DIR/dataset/qa"
VIDEO_ROOT="/home/qhl/code/gaze-mllms/StreamGaze/dataset/videos/egoexolearn/original"
GAZE_VIZ_VIDEO_ROOT="/home/qhl/code/gaze-mllms/StreamGaze/dataset/videos/egoexolearn/viz"  # Gaze visualization videos

# ===== Output =====
WORKDIR="$ROOT_DIR/results/test"
LOGDIR="$WORKDIR/logs"
RESULTS_DIR="$WORKDIR/results"

mkdir -p "$LOGDIR"
mkdir -p "$RESULTS_DIR"

# ===== Choose ONE task only =====
TASK_FILE="past_scene_recall.json"
TASK_NAME="past_scene_recall"

echo "Running test task: $TASK_NAME"

cd ./src

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_name ViSpeak \
    --benchmark_name StreamingBenchGaze_Past_StreamGaze \
    --data_file "$QA_DIR/$TASK_FILE" \
    --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
    --video_root "$VIDEO_ROOT" \
    --use_gaze_instruction \
    --gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT" \
    2>&1 | tee "$LOGDIR/${TASK_NAME}.log"

echo "Done!"