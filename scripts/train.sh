#!/bin/bash
# FunASR-Nano 微调训练脚本
# 使用方法: bash scripts/train.sh

# 配置文件路径
CONFIG_FILE="conf/train_funasr_nano.yaml"

# 数据路径
TRAIN_DATA="data/train.jsonl"
VALID_DATA="data/valid.jsonl"  # 如果没有验证数据，请留空

# 模型路径
MODEL_DIR="models/Fun-ASR-Nano-2512"

# 输出目录
OUTPUT_DIR="exp/funasr_nano_finetune"

# 日志文件
LOG_FILE="${OUTPUT_DIR}/train.log"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 数据格式确认和转换
echo "正在确认数据格式..."
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "错误: 找不到训练数据文件: ${TRAIN_DATA}"
    exit 1
fi

# 数据格式转换（根据需要）
# python scripts/convert_data_format.py --input ${TRAIN_DATA} --output ${TRAIN_DATA}.converted

# 模型目录确认
if [ ! -d "${MODEL_DIR}" ]; then
    echo "警告: 找不到模型目录: ${MODEL_DIR}"
    echo "请下载模型或检查路径。"
    echo "模型下载方法:"
    echo "  python -c \"from funasr import AutoModel; AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512', download_dir='${MODEL_DIR}')\""
    exit 1
fi

# 执行训练
echo "开始训练..."
echo "配置文件: ${CONFIG_FILE}"
echo "训练数据: ${TRAIN_DATA}"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"

# Python路径设置
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/FunASR"

# 执行训练脚本
python src/FunASR/funasr/bin/train_ds.py \
    ++model="${MODEL_DIR}" \
    ++train_data_set_list="${TRAIN_DATA}" \
    ++valid_data_set_list="${VALID_DATA}" \
    ++output_dir="${OUTPUT_DIR}" \
    ++dataset_conf.batch_size=4 \
    ++dataset_conf.batch_type="example" \
    ++dataset_conf.num_workers=4 \
    ++train_conf.max_epoch=10 \
    ++train_conf.log_interval=10 \
    ++train_conf.resume=true \
    ++train_conf.validate_interval=1000 \
    ++train_conf.save_checkpoint_interval=1000 \
    ++train_conf.keep_nbest_models=5 \
    ++train_conf.avg_nbest_model=3 \
    ++train_conf.early_stopping_patience=3 \
    ++optim_conf.lr=0.0001 \
    ++optim_conf.weight_decay=0.0001 \
    ++scheduler_conf.warmup_steps=500 \
    ++llm_conf.use_lora=true \
    ++llm_conf.lora_conf.r=8 \
    ++llm_conf.lora_conf.lora_alpha=16 \
    ++llm_conf.lora_conf.target_modules='["q_proj", "v_proj", "k_proj", "o_proj"]' \
    ++llm_conf.lora_conf.lora_dropout=0.1 \
    ++llm_conf.llm_dtype="fp16" \
    2>&1 | tee ${LOG_FILE}

echo "训练完成。日志文件: ${LOG_FILE}"

