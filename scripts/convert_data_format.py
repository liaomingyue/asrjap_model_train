#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
数据格式转换脚本
将用户的数据格式（key/wav/txt）转换为FunASR期望的格式（source/target）
"""

import json
import os
import sys
import argparse
import librosa


def get_audio_length(wav_path):
    """
    获取音频文件长度（返回帧数）
    假设采样率16kHz，帧移10ms
    """
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        duration = len(y) / sr  # 秒数
        frames = int(duration * 100)  # 10ms单位的帧数
        return frames
    except Exception as e:
        print(f"警告: 读取 {wav_path} 失败: {e}")
        return 100  # 默认值


def get_text_length(text):
    """
    获取文本长度（返回字符数）
    """
    return len(text)


def convert_jsonl(input_file, output_file, calculate_lengths=True):
    """
    转换JSONL文件格式
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        calculate_lengths: 是否计算source_len和target_len
    """
    converted_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 解析JSON
                data = json.loads(line)
                
                # 检查必要字段
                if 'wav' not in data or 'txt' not in data:
                    print(f"警告: 第 {line_num} 行缺少 'wav' 或 'txt' 字段。跳过。")
                    error_count += 1
                    continue
                
                wav_path = data['wav']
                text = data['txt']
                
                # 确认音频文件存在
                if not os.path.exists(wav_path):
                    print(f"警告: 第 {line_num} 行的音频文件未找到: {wav_path}。跳过。")
                    error_count += 1
                    continue
                
                # 转换为新格式
                new_data = {
                    "source": wav_path,
                    "target": text
                }
                
                # 计算source_len和target_len（可选）
                if calculate_lengths:
                    source_len = get_audio_length(wav_path)
                    target_len = get_text_length(text)
                    new_data["source_len"] = source_len
                    new_data["target_len"] = target_len
                
                # 如果存在原始key字段则保留（可选）
                if 'key' in data:
                    new_data["key"] = data['key']
                
                # 以JSONL格式输出
                fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行JSON解析失败: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"错误: 第 {line_num} 行处理过程中发生错误: {e}")
                error_count += 1
                continue
    
    print(f"转换完成:")
    print(f"  成功: {converted_count} 条")
    print(f"  错误: {error_count} 条")
    print(f"  输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='数据格式转换脚本（key/wav/txt -> source/target）'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出JSONL文件路径'
    )
    parser.add_argument(
        '--no-calculate-lengths',
        action='store_true',
        help='不计算source_len和target_len（处理速度更快）'
    )
    
    args = parser.parse_args()
    
    # 确认输入文件存在
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件: {args.input}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 执行转换
    convert_jsonl(
        args.input,
        args.output,
        calculate_lengths=not args.no_calculate_lengths
    )


if __name__ == '__main__':
    main()

