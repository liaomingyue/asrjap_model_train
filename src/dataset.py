"""
Modelscope 数据集加载器
用于从 Modelscope 加载语音识别数据集并进行预处理
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader

# 延迟导入：MsDataset 在实际使用时才进行导入，避免模块加载时的版本冲突
_MsDataset = None

def _get_msdataset():
    """
    延迟导入 MsDataset 的辅助函数
    
    此函数在实际需要使用 MsDataset 时才进行导入，避免在模块加载时
    就触发 modelscope 和 datasets 库的版本冲突问题。
    
    Returns:
        MsDataset 类
        
    Raises:
        ImportError: 当 modelscope 或 datasets 库版本不兼容时抛出
    """
    global _MsDataset
    if _MsDataset is None:
        try:
            from modelscope.msdatasets import MsDataset
            _MsDataset = MsDataset
        except ImportError as e:
            # 检查具体的错误类型，提供更详细的解决方案
            error_detail = str(e)
            if "HubDatasetModuleFactoryWithoutScript" in error_detail or "cannot import name" in error_detail:
                error_msg = (
                    f"modelscope.msdatasets.MsDataset 导入失败: {e}\n"
                    "这通常是由于 modelscope 和 datasets 库的版本不兼容导致的。\n"
                    "请尝试以下解决方案之一：\n"
                    "\n"
                    "方案 1（推荐）：安装兼容的旧版本\n"
                    "  pip install 'datasets<3.0.0' 'modelscope>=1.9.0'\n"
                    "\n"
                    "方案 2：安装兼容的新版本\n"
                    "  pip install 'datasets>=2.14.0' 'modelscope>=1.15.0'\n"
                    "\n"
                    "方案 3：重新安装兼容版本组合\n"
                    "  pip uninstall modelscope datasets -y\n"
                    "  pip install 'datasets==2.14.7' 'modelscope==1.15.0'\n"
                    "\n"
                    "如果问题仍然存在，请检查当前安装的版本：\n"
                    "  pip show modelscope datasets"
                )
            else:
                error_msg = (
                    f"modelscope.msdatasets.MsDataset 导入失败: {e}\n"
                    "请确保已正确安装 modelscope 库：\n"
                    "  pip install modelscope"
                )
            logging.error(error_msg)
            raise ImportError(error_msg) from e
    return _MsDataset


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
        
        # 将 cache_dir 转换为绝对路径，并确保目录存在
        if cache_dir:
            cache_dir = os.path.abspath(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            logging.info(f"使用缓存目录: {cache_dir}")
        else:
            # 如果未指定 cache_dir，使用默认值
            cache_dir = os.path.abspath("data")
            os.makedirs(cache_dir, exist_ok=True)
            logging.info(f"使用默认缓存目录: {cache_dir}")
        
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.frontend = frontend
        self.model = model
        self.kwargs = kwargs
        
        # 加载数据集
        # 注意：MsDataset.load() 不支持 streaming 参数，因此不传递该参数
        logging.info(f"正在加载数据集: {dataset_name}, 子集: {subset_name}, 分割: {split}")
        MsDataset = _get_msdataset()  # 延迟导入，避免版本冲突
        
        # MsDataset.load() 不支持 streaming 参数，因此不传递
        # 数据集的流式处理由后续的数据访问方式决定
        try:
            self.dataset = MsDataset.load(
                dataset_name,
                subset_name=subset_name,
                split=split,
                cache_dir=cache_dir
            )
        except (TypeError, FileNotFoundError) as e:
            error_msg = str(e)
            # 如果出现参数错误，尝试不传递 subset_name
            if "unexpected keyword argument" in error_msg:
                logging.warning(f"加载数据集时出现参数错误: {e}，尝试使用简化参数...")
                try:
                    self.dataset = MsDataset.load(
                        dataset_name,
                        split=split,
                        cache_dir=cache_dir
                    )
                except Exception as e2:
                    logging.warning(f"使用简化参数仍然失败: {e2}，尝试不指定 cache_dir...")
                    # 最后尝试不指定 cache_dir
                    self.dataset = MsDataset.load(
                        dataset_name,
                        subset_name=subset_name,
                        split=split
                    )
            # 如果出现文件路径错误，尝试不指定 cache_dir
            elif "No such file or directory" in error_msg or "FileNotFoundError" in str(type(e)):
                logging.warning(f"缓存目录路径错误: {e}，尝试不指定 cache_dir...")
                try:
                    self.dataset = MsDataset.load(
                        dataset_name,
                        subset_name=subset_name,
                        split=split
                    )
                except Exception as e2:
                    logging.warning(f"不指定 cache_dir 仍然失败: {e2}，尝试最简参数...")
                    # 最后尝试最简参数
                    self.dataset = MsDataset.load(
                        dataset_name,
                        split=split
                    )
            else:
                # 其他错误直接抛出
                raise
        
        # 如果不是流式加载，转换为列表
        if not streaming:
            logging.info("正在将数据集转换为列表...")
            self.data_list = list(self.dataset)
            logging.info(f"数据集加载完成，共 {len(self.data_list)} 条数据")
        else:
            self.data_list = None
            logging.info("使用流式数据集加载模式（数据将在访问时动态加载）")
    
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
        # 尝试多种可能的字段名，以适配不同的数据集格式（支持大小写不敏感和带冒号的字段名）
        audio_path = None
        text = None
        
        # 获取所有可用的字段名（转换为小写以便比较）
        if isinstance(raw_data, dict):
            available_keys_lower = {k.lower(): k for k in raw_data.keys()}
        else:
            available_keys_lower = {}
        
        # 尝试多种音频字段名（大小写不敏感）
        audio_keys = [
            'audio', 'audio:file', 'audio_file', 'audio_path', 'audio:path',
            'path', 'file', 'wav', 'wav_file', 'audio:FILE', 'Audio:FILE'
        ]
        for key in audio_keys:
            key_lower = key.lower()
            # 直接匹配
            if key in raw_data and raw_data[key] is not None:
                audio_path = raw_data[key]
                break
            # 大小写不敏感匹配
            elif key_lower in available_keys_lower:
                actual_key = available_keys_lower[key_lower]
                if raw_data[actual_key] is not None:
                    audio_path = raw_data[actual_key]
                    break
        
        # 如果 audio_path 是字典，尝试从中提取路径
        if isinstance(audio_path, dict):
            audio_path = audio_path.get('path', None) or audio_path.get('file', None)
        
        # 尝试多种文本字段名（大小写不敏感）
        text_keys = [
            'text', 'sentence', 'transcription', 'transcript', 'label', 
            'target', 'output', 'text:label', 'Text:LABEL'
        ]
        for key in text_keys:
            key_lower = key.lower()
            # 直接匹配
            if key in raw_data and raw_data[key] is not None:
                text = raw_data[key]
                break
            # 大小写不敏感匹配
            elif key_lower in available_keys_lower:
                actual_key = available_keys_lower[key_lower]
                if raw_data[actual_key] is not None:
                    text = raw_data[actual_key]
                    break
        
        # 如果仍然找不到必要字段，记录详细信息并跳过
        if audio_path is None or text is None:
            available_keys = list(raw_data.keys()) if isinstance(raw_data, dict) else "非字典类型"
            logging.warning(
                f"数据 {idx} 缺少必要字段: audio={audio_path}, text={text}, "
                f"可用字段: {available_keys}"
            )
            # 返回 None，由 collate_fn 过滤
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
        speeches = []
        speech_lengths_list = []
        
        for b in batch:
            if len(b.get("speech", [])) > 0:
                speeches.append(b["speech"])
                # speech_lengths の処理：テンソルの場合は形状を確認して適切に処理
                sl = b.get("speech_lengths", None)
                if sl is not None:
                    if isinstance(sl, torch.Tensor):
                        if len(sl.shape) > 0 and sl.shape[0] > 0:
                            # [批次, ...] の形状の場合、最初の要素を取得
                            speech_lengths_list.append(sl[0])
                        else:
                            # スカラーの場合
                            speech_lengths_list.append(sl)
                    else:
                        # リストやその他の型の場合
                        speech_lengths_list.append(torch.tensor(sl[0] if isinstance(sl, (list, tuple)) else sl, dtype=torch.int64))
                else:
                    # speech_lengths が存在しない場合、speech の長さから推定
                    s = b["speech"]
                    if len(s.shape) >= 2:
                        speech_lengths_list.append(torch.tensor(s.shape[0] if len(s.shape) == 2 else s.shape[1], dtype=torch.int64))
                    else:
                        speech_lengths_list.append(torch.tensor(0, dtype=torch.int64))
        
        if len(speeches) > 0:
            try:
                # 检查所有 speech テンソルの形状
                shapes = [s.shape for s in speeches]
                feature_dims = [s.shape[-1] if len(s.shape) > 1 else 1 for s in speeches]
                
                # 如果特徴次元が異なる場合、最大次元に合わせてパディング
                max_feature_dim = max(feature_dims)
                min_feature_dim = min(feature_dims)
                
                if max_feature_dim != min_feature_dim:
                    logging.warning(
                        f"检测到批次中 speech 特征维度不一致: {feature_dims}，"
                        f"将统一到最大维度: {max_feature_dim}"
                    )
                    # 各 speech テンソルを最大特徴次元にパディング
                    padded_speeches = []
                    for s in speeches:
                        current_feature_dim = s.shape[-1] if len(s.shape) > 1 else 1
                        if current_feature_dim < max_feature_dim:
                            # 特徴次元をパディング
                            pad_size = max_feature_dim - current_feature_dim
                            if len(s.shape) == 2:
                                # [时间, 特征] -> [时间, 最大特征]
                                padding = torch.zeros(s.shape[0], pad_size, dtype=s.dtype, device=s.device)
                                s = torch.cat([s, padding], dim=1)
                            elif len(s.shape) == 3:
                                # [批次, 时间, 特征] -> [批次, 时间, 最大特征]
                                # 通常は [1, 时间, 特征] の形状
                                if s.shape[0] == 1:
                                    # [1, 时间, 特征] -> [时间, 最大特征]
                                    s = s[0]  # [时间, 特征]
                                    padding = torch.zeros(s.shape[0], pad_size, dtype=s.dtype, device=s.device)
                                    s = torch.cat([s, padding], dim=1)
                                else:
                                    padding = torch.zeros(s.shape[0], s.shape[1], pad_size, dtype=s.dtype, device=s.device)
                                    s = torch.cat([s, padding], dim=2)
                            else:
                                logging.error(f"不支持的 speech 形状: {s.shape}")
                                raise ValueError(f"不支持的 speech 形状: {s.shape}")
                        padded_speeches.append(s)
                    speeches = padded_speeches
                
                # すべての speech テンソルを [时间, 特征] の形状に統一
                speeches_for_pad = []
                for s in speeches:
                    if len(s.shape) == 3:
                        # [批次, 时间, 特征] -> [时间, 特征]
                        if s.shape[0] == 1:
                            s = s[0]  # [时间, 特征]
                        else:
                            # 複数のバッチがある場合、最初のものを使用（通常は発生しない）
                            logging.warning(f"意外的 speech 形状: {s.shape}，使用第一个批次")
                            s = s[0]
                    elif len(s.shape) == 2:
                        # すでに [时间, 特征] の形式
                        pass
                    else:
                        raise ValueError(f"不支持的 speech 形状: {s.shape}")
                    speeches_for_pad.append(s)
                
                # すべての speech テンソルが同じ特徴次元を持つことを確認
                final_feature_dims = [s.shape[-1] for s in speeches_for_pad]
                if len(set(final_feature_dims)) > 1:
                    raise ValueError(f"特征维度仍然不一致: {final_feature_dims}, 形状: {[s.shape for s in speeches_for_pad]}")
                
                # pad_sequence を使用して時間次元をパディング
                # 入力は [时间, 特征] のリストで、出力は [批次, 时间, 特征]
                speech = torch.nn.utils.rnn.pad_sequence(
                    speeches_for_pad, batch_first=True, padding_value=0.0
                )
                
                # speech_lengths をスタック
                if len(speech_lengths_list) > 0:
                    speech_lengths = torch.stack(speech_lengths_list)
                else:
                    speech_lengths = torch.tensor([], dtype=torch.int64)
                
            except Exception as e:
                logging.error(f"合并 speech 时发生错误: {e}")
                logging.error(f"Speech 形状: {[s.shape for s in speeches]}")
                logging.error(f"Speech lengths: {speech_lengths_list}")
                raise
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

