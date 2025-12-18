"""
语音模型微调训练脚本
使用 Modelscope 数据集进行模型训练
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from funasr import AutoModel
from funasr.train_utils.device_funcs import to_device
from src.dataset import create_dataset
from src.model import FunASRNano

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        logger.info(f"分布式训练初始化: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank
    else:
        logger.info("单GPU训练模式")
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(checkpoint_path, checkpoint_data, max_retries=3):
    """
    保存检查点，包含错误处理和重试机制
    
    Args:
        checkpoint_path: 检查点保存路径
        checkpoint_data: 要保存的数据字典
        max_retries: 最大重试次数
        
    Returns:
        bool: 保存是否成功
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(checkpoint_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 检查磁盘空间（如果可能）
    try:
        stat = shutil.disk_usage(output_dir if output_dir else os.getcwd())
        free_space_gb = stat.free / (1024 ** 3)
        if free_space_gb < 1.0:  # 小于 1GB
            logger.warning(f"磁盘空间不足: {free_space_gb:.2f} GB 可用")
    except Exception as e:
        logger.warning(f"无法检查磁盘空间: {e}")
    
    # 尝试保存，带重试机制
    for attempt in range(max_retries):
        try:
            # 先保存到临时文件，然后重命名（原子操作）
            temp_path = checkpoint_path + ".tmp"
            
            # 如果临时文件存在，先删除
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # 保存到临时文件
            torch.save(checkpoint_data, temp_path)
            
            # 如果目标文件存在，先删除
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            
            # 重命名为最终文件（原子操作）
            os.rename(temp_path, checkpoint_path)
            
            logger.info(f"成功保存检查点: {checkpoint_path}")
            return True
            
        except RuntimeError as e:
            error_msg = str(e)
            if "file write failed" in error_msg or "unexpected pos" in error_msg:
                logger.error(f"保存检查点时发生文件写入错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # 等待一段时间后重试
                    import time
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    logger.error(f"保存检查点失败，已达到最大重试次数: {checkpoint_path}")
                    return False
            else:
                # 其他运行时错误直接抛出
                raise
        except Exception as e:
            logger.error(f"保存检查点时发生未知错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
                continue
            else:
                logger.error(f"保存检查点失败: {checkpoint_path}")
                return False
    
    return False


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    accumulation_steps=1,
    max_grad_norm=1.0
):
    """
    训练一个epoch
    
    Args:
        model: 模型实例
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        accumulation_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪阈值
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        # 将数据移动到设备
        batch = to_device(batch, device)
        
        # 前向传播
        loss, stats, weight = model(**batch)
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 优化器步进
            optimizer.step()
            optimizer.zero_grad()
        
        # 统计信息
        total_loss += stats.get("loss", 0.0).item() * weight.item()
        total_acc += stats.get("acc", 0.0) * weight.item()
        num_batches += weight.item()
        
        # 打印进度
        if batch_idx % 10 == 0:
            current_loss = total_loss / num_batches if num_batches > 0 else 0.0
            current_acc = total_acc / num_batches if num_batches > 0 else 0.0
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}, "
                f"Loss: {current_loss:.4f}, Acc: {current_acc:.4f}"
            )
    
    # 处理剩余的梯度
    if num_batches % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
    
    logger.info(f"Epoch {epoch} 完成 - 平均损失: {avg_loss:.4f}, 平均准确率: {avg_acc:.4f}")
    
    return avg_loss, avg_acc


def validate(model, dataloader, device, epoch):
    """
    验证模型
    
    Args:
        model: 模型实例
        dataloader: 数据加载器
        device: 设备
        epoch: 当前epoch
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            # 将数据移动到设备
            batch = to_device(batch, device)
            
            # 前向传播
            loss, stats, weight = model(**batch)
            
            # 统计信息
            total_loss += stats.get("loss", 0.0).item() * weight.item()
            total_acc += stats.get("acc", 0.0) * weight.item()
            num_batches += weight.item()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
    
    logger.info(f"Epoch {epoch} 验证 - 平均损失: {avg_loss:.4f}, 平均准确率: {avg_acc:.4f}")
    
    return avg_loss, avg_acc


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="语音模型微调训练")
    
    # 数据集参数
    parser.add_argument("--dataset_name", type=str, 
                       default="speech_asr/speech_asr_aishell1_subset",
                       help="数据集名称")
    parser.add_argument("--subset_name", type=str, default="default",
                       help="子集名称")
    parser.add_argument("--train_split", type=str, default="train",
                       help="训练集分割")
    parser.add_argument("--val_split", type=str, default="validation",
                       help="验证集分割")
    parser.add_argument("--cache_dir", type=str, default="data",
                       help="缓存目录")
    parser.add_argument("--streaming", action="store_true",
                       help="使用流式数据集加载")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True,
                       help="模型路径或名称")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="学习率")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="梯度裁剪阈值")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="保存检查点的步数")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="评估步数")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="训练设备")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载工作进程数")
    
    args = parser.parse_args()
    
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()
    
    # 设置设备
    if args.device == "cuda":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info(f"正在加载模型: {args.model}")
    model, kwargs = FunASRNano.from_pretrained(model=args.model)
    model = model.to(device)
    
    # 分布式训练包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model_module = model.module
    else:
        model_module = model
    
    # 获取 tokenizer 和 frontend
    tokenizer = kwargs.get("tokenizer", None)
    frontend = kwargs.get("frontend", None)
    
    if tokenizer is None or frontend is None:
        logger.warning("未找到 tokenizer 或 frontend，将使用默认配置")
    
    # 从 kwargs 中移除已明确传递的参数，避免重复参数错误
    dataset_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ["model", "tokenizer", "frontend"]}
    
    # 创建数据集
    logger.info("正在创建训练数据集...")
    train_dataset = create_dataset(
        dataset_name=args.dataset_name,
        subset_name=args.subset_name,
        split=args.train_split,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
        tokenizer=tokenizer,
        frontend=frontend,
        model=model_module,
        **dataset_kwargs
    )
    
    train_dataloader = train_dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 创建验证数据集（如果存在）
    val_dataloader = None
    if args.val_split:
        try:
            logger.info("正在创建验证数据集...")
            val_dataset = create_dataset(
                dataset_name=args.dataset_name,
                subset_name=args.subset_name,
                split=args.val_split,
                streaming=args.streaming,
                cache_dir=args.cache_dir,
                tokenizer=tokenizer,
                frontend=frontend,
                model=model_module,
                **dataset_kwargs
            )
            val_dataloader = val_dataset.get_dataloader(
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
        except Exception as e:
            logger.warning(f"无法创建验证数据集: {e}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练循环
    logger.info("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"开始训练 Epoch {epoch}/{args.epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model,
            train_dataloader,
            optimizer,
            device,
            epoch,
            args.accumulation_steps,
            args.max_grad_norm
        )
        
        # 验证
        if val_dataloader is not None:
            val_loss, val_acc = validate(model, val_dataloader, device, epoch)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"保存最佳模型 (验证损失: {val_loss:.4f})")
                if rank == 0:  # 只在主进程保存
                    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': model_module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                    }
                    if not save_checkpoint(checkpoint_path, checkpoint_data):
                        logger.error("保存最佳模型失败，但训练将继续进行")
        
        # 定期保存检查点
        if epoch % (args.save_steps // len(train_dataloader) + 1) == 0 and rank == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }
            if not save_checkpoint(checkpoint_path, checkpoint_data):
                logger.error(f"保存检查点失败: {checkpoint_path}")
    
    logger.info("训练完成！")
    
    # 保存最终模型
    if rank == 0:
        final_path = os.path.join(args.output_dir, "final_model.pt")
        checkpoint_data = {
            'model_state_dict': model_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if not save_checkpoint(final_path, checkpoint_data):
            logger.error("保存最终模型失败")
    
    # 清理分布式训练
    cleanup_distributed()


if __name__ == "__main__":
    main()

