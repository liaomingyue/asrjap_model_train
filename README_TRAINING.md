# FunASR-Nano 微调训练指南

本文档介绍FunASR-Nano模型的微调（fine-tuning）方法。

## 目录

1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [模型准备](#模型准备)
4. [训练执行](#训练执行)
5. [设置自定义](#设置自定义)
6. [故障排除](#故障排除)

## 环境准备

### 安装必要的包

```bash
# 安装FunASR
cd src/FunASR
pip install -e .

# 其他依赖包
pip install torch torchaudio librosa transformers peft
```

### Python路径设置

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/FunASR"
```

## 数据准备

### 数据格式

训练数据为JSONL格式，每行代表一个样本。

**原始格式（用户的数据格式）:**
```json
{"key": "BAC009S0764W0126", "wav": "/path/to/audio.wav", "txt": "因此土地储备至关重要"}
```

**FunASR期望的格式:**
```json
{"source": "/path/to/audio.wav", "target": "因此土地储备至关重要", "source_len": 1000, "target_len": 10}
```

### 数据格式转换

可以使用提供的脚本转换数据格式：

```bash
python scripts/convert_data_format.py \
    --input data/train.jsonl \
    --output data/train_converted.jsonl
```

使用`--no-calculate-lengths`选项可以跳过`source_len`和`target_len`的计算，加快处理速度（但可能导致数据过滤无法正常工作）。

### 数据目录结构

```
data/
├── train.jsonl          # 训练数据
└── valid.jsonl          # 验证数据（可选）
```

## 模型准备

### 模型下载

下载FunASR-Nano模型：

```python
from funasr import AutoModel

# 下载模型
model = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    download_dir="models/Fun-ASR-Nano-2512"
)
```

或者，也可以直接从ModelScope或HuggingFace下载。

### 模型目录结构

```
models/Fun-ASR-Nano-2512/
├── config.yaml          # 模型配置文件
├── model.pt             # 模型参数
├── tokenizer/           # 分词器目录
└── ...
```

## 训练执行

### 方法1: 使用Shell脚本

```bash
bash scripts/train.sh
```

可以编辑脚本中的变量来更改数据路径或模型路径。

### 方法2: 使用Python脚本

```bash
python scripts/train.py \
    --model_dir models/Fun-ASR-Nano-2512 \
    --train_data data/train.jsonl \
    --valid_data data/valid.jsonl \
    --output_dir exp/funasr_nano_finetune \
    --batch_size 4 \
    --max_epoch 10 \
    --learning_rate 0.0001
```

### 方法3: 直接从命令行执行

```bash
python src/FunASR/funasr/bin/train_ds.py \
    ++model="models/Fun-ASR-Nano-2512" \
    ++train_data_set_list="data/train.jsonl" \
    ++valid_data_set_list="data/valid.jsonl" \
    ++output_dir="exp/funasr_nano_finetune" \
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
    ++llm_conf.llm_dtype="fp16"
```

## 设置自定义

### 主要设置参数

#### 训练设置 (`train_conf`)

- `max_epoch`: 最大epoch数（默认: 10）
- `log_interval`: 日志输出间隔（默认: 10）
- `validate_interval`: 验证间隔（步数，默认: 1000）
- `save_checkpoint_interval`: 检查点保存间隔（步数，默认: 1000）
- `early_stopping_patience`: 早停耐心值（epoch数，默认: 3）

#### 数据集设置 (`dataset_conf`)

- `batch_size`: 批次大小（默认: 4）
- `batch_type`: 批次类型（`example` 或 `length`，默认: `example`）
- `num_workers`: 数据加载器工作进程数（默认: 4）
- `max_source_length`: 最大源长度（默认: 2000）
- `max_target_length`: 最大目标长度（默认: 200）

#### 优化器设置 (`optim_conf`)

- `lr`: 学习率（默认: 0.0001）
- `weight_decay`: 权重衰减（默认: 0.0001）

#### LoRA设置 (`llm_conf.lora_conf`)

- `r`: LoRA rank（默认: 8）
- `lora_alpha`: LoRA alpha（默认: 16）
- `target_modules`: 目标模块（默认: `["q_proj", "v_proj", "k_proj", "o_proj"]`）
- `lora_dropout`: LoRA dropout率（默认: 0.1）

### 编辑配置文件

可以编辑`conf/train_funasr_nano.yaml`来自定义设置。

## 训练结果

训练完成后，结果将保存在以下目录：

```
exp/funasr_nano_finetune/
├── train.log              # 训练日志
├── config.yaml            # 使用的配置文件
├── checkpoint.pth         # 检查点
├── model.pt               # 最终模型
└── ...
```

## 故障排除

### 常见问题

#### 1. 内存不足错误

- 减小批次大小（将`batch_size`调小）
- 使用梯度累积（增加`accum_grad`）
- 使用混合精度训练（`llm_dtype: fp16`）

#### 2. 数据加载错误

- 检查音频文件路径是否正确
- 检查JSONL文件格式是否正确
- 检查`source_len`和`target_len`是否已计算

#### 3. 模型加载错误

- 检查模型目录路径是否正确
- 检查模型是否正确下载
- 检查`config.yaml`是否存在

#### 4. CUDA out of memory

- 减小批次大小
- 使用`llm_dtype: fp16`或`bf16`
- 设置`activation_checkpoint: true`

### 日志查看

训练日志保存在`exp/funasr_nano_finetune/train.log`。如果发生错误，请查看此文件。

## 参考资料

- [FunASR官方文档](https://github.com/alibaba-damo-academy/FunASR)
- [FunASR-Nano模型页面](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)

## 许可证

本项目遵循FunASR的许可证。

