"""
Modelscope 数据集加载器
用于从 Modelscope 加载语音识别数据集并进行预处理
"""

import logging
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from modelscope.msdatasets import MsDataset


class ModelscopeASRDataset(Dataset):
    """Modelscope 语音识别数据集类"""
    
    def __init__(
        self,
        dataset_name: str,
        subset_name: str = "default",
        split: str = "train",
        streaming: bool = True,
        cache_dir: str = "data",
        tokenizer=None,
        frontend=None,
        model=None,
        **kwargs
    ):
        """
        初始化数据集
        
        Args:
            dataset_name: 数据集名称，例如 'speech_asr/speech_asr_aishell1_subset'
            subset_name: 子集名称，默认为 'default'
            split: 数据集分割，'train' 或 'test'
            streaming: 是否使用流式加载
            cache_dir: 缓存目录
            tokenizer: 分词器
            frontend: 前端处理器
            model: 模型实例
            **kwargs: 其他参数
        """
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.split = split
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.frontend = frontend
        self.model = model
        self.kwargs = kwargs
        
        # 加载数据集
        logging.info(f"正在加载数据集: {dataset_name}, 子集: {subset_name}, 分割: {split}")
        self.dataset = MsDataset.load(
            dataset_name,
            subset_name=subset_name,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir
        )
        
        # 如果不是流式加载，转换为列表
        if not streaming:
            logging.info("正在将数据集转换为列表...")
            self.data_list = list(self.dataset)
            logging.info(f"数据集加载完成，共 {len(self.data_list)} 条数据")
        else:
            self.data_list = None
            logging.info("使用流式数据集加载模式")
    
    def __len__(self):
        """返回数据集大小"""
        if self.data_list is not None:
            return len(self.data_list)
        else:
            # 流式数据集无法直接获取长度
            return None
    
    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
            
        Returns:
            处理后的数据字典
        """
        # 获取原始数据
        if self.data_list is not None:
            raw_data = self.data_list[idx]
        else:
            # 流式数据集需要特殊处理
            raw_data = list(self.dataset.skip(idx).take(1))[0]
        
        # 将原始数据转换为模型所需的格式
        # 假设原始数据包含 'audio' 和 'text' 字段
        # 根据实际数据集结构调整
        audio_path = raw_data.get('audio', None) or raw_data.get('audio:FILE', None)
        text = raw_data.get('text', None) or raw_data.get('sentence', None) or raw_data.get('transcription', None)
        
        if audio_path is None or text is None:
            logging.warning(f"数据 {idx} 缺少必要字段: audio={audio_path}, text={text}")
            # 返回空数据或跳过
            return None
        
        # 构建模型所需的数据格式
        data_item = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"语音转写：<|startofspeech|>!{audio_path}<|endofspeech|>",
            },
            {"role": "assistant", "content": text},
        ]
        
        # 使用模型的数据加载方法处理
        if self.model is not None and self.tokenizer is not None and self.frontend is not None:
            contents = self.model.data_template(data_item)
            output = self.model.data_load_speech(
                contents, 
                self.tokenizer, 
                self.frontend, 
                meta_data={}, 
                **self.kwargs
            )
            return output
        else:
            # 返回原始格式
            return {
                "data": data_item,
                "audio_path": audio_path,
                "text": text
            }
    
    def get_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        **kwargs
    ):
        """
        创建 DataLoader
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            collate_fn: 批处理函数
            **kwargs: 其他 DataLoader 参数
            
        Returns:
            DataLoader 实例
        """
        if collate_fn is None:
            collate_fn = self._default_collate_fn
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs
        )
    
    def _default_collate_fn(self, batch):
        """
        默认的批处理函数
        
        Args:
            batch: 批次数据列表
            
        Returns:
            批处理后的数据
        """
        # 过滤掉 None 值
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        # 如果数据已经由模型处理过，直接返回
        if isinstance(batch[0], dict) and "input_ids" in batch[0]:
            # 合并批次数据
            return self._merge_batch(batch)
        else:
            return batch
    
    def _merge_batch(self, batch):
        """
        合并批次数据
        
        Args:
            batch: 批次数据列表
            
        Returns:
            合并后的批次数据
        """
        # 合并 speech
        speeches = [b["speech"] for b in batch if len(b.get("speech", [])) > 0]
        if len(speeches) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(
                speeches, batch_first=True, padding_value=0.0
            )
            speech_lengths = torch.stack([
                b["speech_lengths"][0] if len(b.get("speech", [])) > 0 
                else torch.tensor([0], dtype=torch.int64) 
                for b in batch
            ])
        else:
            speech = []
            speech_lengths = []
        
        # 合并其他张量
        merged = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": torch.cat([b["fbank_mask"] for b in batch], dim=0),
            "fbank_beg": torch.cat([b["fbank_beg"] for b in batch], dim=0),
            "fake_token_len": torch.cat([b["fake_token_len"] for b in batch], dim=0),
            "input_ids": torch.cat([b["input_ids"] for b in batch], dim=0),
            "attention_mask": torch.cat([b["attention_mask"] for b in batch], dim=0),
            "labels_ids": torch.cat([b["labels_ids"] for b in batch], dim=0),
            "source_ids": torch.cat([b["source_ids"] for b in batch], dim=0),
            "target_ids": torch.cat([b["target_ids"] for b in batch], dim=0),
        }
        
        return merged


def create_dataset(
    dataset_name: str,
    subset_name: str = "default",
    split: str = "train",
    streaming: bool = True,
    cache_dir: str = "data",
    tokenizer=None,
    frontend=None,
    model=None,
    **kwargs
) -> ModelscopeASRDataset:
    """
    创建 Modelscope 数据集实例的便捷函数
    
    Args:
        dataset_name: 数据集名称
        subset_name: 子集名称
        split: 数据集分割
        streaming: 是否使用流式加载
        cache_dir: 缓存目录
        tokenizer: 分词器
        frontend: 前端处理器
        model: 模型实例
        **kwargs: 其他参数
        
    Returns:
        ModelscopeASRDataset 实例
    """
    return ModelscopeASRDataset(
        dataset_name=dataset_name,
        subset_name=subset_name,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        frontend=frontend,
        model=model,
        **kwargs
    )

