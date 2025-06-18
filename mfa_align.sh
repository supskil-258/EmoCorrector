#!/bin/bash
set -e  # 如果发生错误则退出

# 并行处理的任务数量，可以根据实际情况调整
NUM_JOB=${NUM_JOB:-128}
echo "| 使用 ${NUM_JOB} 核心运行 MFA 对齐。"

# 定义基础目录和 MFA 模型路径
BASE_DIR=assets/mfa_temp
MODEL_MFA=checkpoints/mfa_model.zip  # 确保模型路径正确

# 定义输入和输出目录
INPUT_DIR=$BASE_DIR/mfa_inputs
OUTPUT_TMP_DIR=$BASE_DIR/mfa_outputs_tmp
TMP_ALIGN_DIR=$BASE_DIR/mfa_tmp
OUTPUT_DIR=$BASE_DIR/mfa_outputs

mkdir -p $INPUT_DIR
mkdir -p $OUTPUT_TMP_DIR
mkdir -p $TMP_ALIGN_DIR
mkdir -p $OUTPUT_DIR

DICT_FILE=dataset/processed/azure/noisy_mfa_dict.txt  # 使用指定的字典文件

echo "| 对齐所有音频文件..."
mfa align \
    $INPUT_DIR/ \
    $DICT_FILE \
    $MODEL_MFA \
    $OUTPUT_TMP_DIR/ \
    -t $TMP_ALIGN_DIR/ \
    -j $NUM_JOB

echo "| 同步对齐结果到最终输出目录..."
find $OUTPUT_TMP_DIR -name "*.TextGrid" -print0 | xargs -0 -I {} cp {} $OUTPUT_DIR/

echo "| 清理临时目录..."
rm -rf $OUTPUT_TMP_DIR $TMP_ALIGN_DIR

echo "| MFA 对齐完成。"
