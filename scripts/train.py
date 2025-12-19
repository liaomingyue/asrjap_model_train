#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
FunASR-Nano 微调训练脚本（Python版）
使用方法: python scripts/train.py
"""

import os
import sys
import argparse
import logging

# 添加FunASR路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'FunASR'))

from funasr.bin.train_ds import main


def setup_logging(log_file):
    """日志设置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main_train():
    """训练主函数"""
    parser = argparse.ArgumentParser(description='FunASR-Nano 微调训练')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models/Fun-ASR-Nano-2512',
        help='模型目录路径'
    )
    parser.add_argument(
        '--train_data',
        type=str,
        default='data/train.jsonl',
        help='训练数据文件路径'
    )
    parser.add_argument(
        '--valid_data',
        type=str,
        default='data/valid.jsonl',
        help='验证数据文件路径（可选）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='exp/funasr_nano_finetune',
        help='输出目录路径'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='conf/train_funasr_nano.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批次大小'
    )
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=10,
        help='最大epoch数'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='学习率'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 日志文件设置
    log_file = os.path.join(args.output_dir, 'train.log')
    setup_logging(log_file)
    
    # 数据文件确认
    if not os.path.exists(args.train_data):
        logging.error(f"找不到训练数据文件: {args.train_data}")
        sys.exit(1)
    
    if args.valid_data and not os.path.exists(args.valid_data):
        logging.warning(f"找不到验证数据文件: {args.valid_data}")
        args.valid_data = None
    
    # 模型目录确认
    if not os.path.exists(args.model_dir):
        logging.error(f"找不到模型目录: {args.model_dir}")
        logging.info("请下载模型:")
        logging.info(f"  python -c \"from funasr import AutoModel; AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512', download_dir='{args.model_dir}')\"")
        sys.exit(1)
    
    # 训练参数设置
    kwargs = {
        'model': args.model_dir,
        'train_data_set_list': args.train_data,
        'output_dir': args.output_dir,
        'dataset_conf': {
            'batch_size': args.batch_size,
            'batch_type': 'example',
            'num_workers': 4,
        },
        'train_conf': {
            'max_epoch': args.max_epoch,
            'log_interval': 10,
            'resume': True,
            'validate_interval': 1000,
            'save_checkpoint_interval': 1000,
            'keep_nbest_models': 5,
            'avg_nbest_model': 3,
            'early_stopping_patience': 3,
        },
        'optim_conf': {
            'lr': args.learning_rate,
            'weight_decay': 0.0001,
        },
        'scheduler_conf': {
            'warmup_steps': 500,
        },
        'llm_conf': {
            'use_lora': True,
            'lora_conf': {
                'r': 8,
                'lora_alpha': 16,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                'lora_dropout': 0.1,
            },
            'llm_dtype': 'fp16',
        },
    }
    
    if args.valid_data:
        kwargs['valid_data_set_list'] = args.valid_data
    
    # 如果配置文件存在则读取
    if os.path.exists(args.config_file):
        logging.info(f"正在读取配置文件: {args.config_file}")
        # 可以在这里添加读取配置文件并更新kwargs的处理
        # 当前直接使用命令行参数和kwargs
    
    logging.info("开始训练...")
    logging.info(f"模型目录: {args.model_dir}")
    logging.info(f"训练数据: {args.train_data}")
    logging.info(f"输出目录: {args.output_dir}")
    
    # 执行训练
    try:
        main(**kwargs)
        logging.info("训练已正常完成。")
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main_train()

