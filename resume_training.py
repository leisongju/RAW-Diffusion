#!/usr/bin/env python3
"""
从检查点恢复训练的脚本
支持多种恢复方式：
1. 自动检测last.ckpt
2. 指定检查点路径
3. 选择最新的检查点
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
import lightning.pytorch.callbacks as callbacks
from aim.pytorch_lightning import AimLogger

from rawdiffusion.datasets.dataset_factory import create_dataset
from rawdiffusion.gaussian_diffusion_factory import create_gaussian_diffusion
from rawdiffusion.resample import create_named_schedule_sampler
from rawdiffusion.utils import get_output_path
from rawdiffusion.config import mod_config
from train_pdraw import PDRAWDiffusionModule


def find_latest_checkpoint(experiment_folder):
    """查找最新的检查点文件"""
    checkpoint_dir = os.path.join(experiment_folder, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not ckpt_files:
        return None
    
    # 按修改时间排序，选择最新的
    ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return os.path.join(checkpoint_dir, ckpt_files[0])


def resume_training(cfg: DictConfig, checkpoint_path: str = None):
    """从检查点恢复训练"""
    mod_config(cfg)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.general.seed)

    aim_logger = AimLogger(
        experiment="pdraw_diffusion",
        train_metric_prefix=None,
        test_metric_prefix=None,
        val_metric_prefix=None,
    )

    experiment_folder = get_output_path(cfg)
    print(f"experiment_folder: {experiment_folder}")

    print("creating data loader...")
    data_train = create_dataset(
        **cfg.dataset.train, seed=cfg.general.seed, patch_size=cfg.general.image_size
    )
    data_val = create_dataset(
        **cfg.dataset.val,
        seed=cfg.general.seed,
        patch_size=cfg.general.image_size,
        permutate_once=True,
    )

    raw_module = PDRAWDiffusionModule(experiment_folder=experiment_folder, **cfg)

    trainer_callbacks = [
        callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    if cfg.general.checkpoint:
        checkpoint_path_dir = os.path.join(experiment_folder, "checkpoints")
        checkpoint_cb = callbacks.ModelCheckpoint(
            dirpath=checkpoint_path_dir,
            save_last=True,
            every_n_train_steps=cfg.general.save_interval,
            enable_version_counter=False,
        )
        trainer_callbacks.append(checkpoint_cb)
        print("checkpoint", experiment_folder)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=cfg.general.max_steps,
        logger=aim_logger,
        callbacks=trainer_callbacks,
        enable_checkpointing=cfg.general.checkpoint,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epoch,
        limit_train_batches=1000,
    )

    # 确定检查点路径
    if checkpoint_path is None:
        # 自动检测last.ckpt
        last_ckpt_path = os.path.join(experiment_folder, "checkpoints", "last.ckpt")
        if os.path.exists(last_ckpt_path):
            checkpoint_path = last_ckpt_path
            print(f"自动检测到last.ckpt: {checkpoint_path}")
        else:
            # 查找最新检查点
            latest_ckpt = find_latest_checkpoint(experiment_folder)
            if latest_ckpt:
                checkpoint_path = latest_ckpt
                print(f"使用最新检查点: {checkpoint_path}")
            else:
                print("未找到检查点文件，开始新训练")
                trainer.fit(raw_module, data_train, data_val)
                return

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"从检查点恢复训练: {checkpoint_path}")
        trainer.fit(raw_module, data_train, data_val, ckpt_path=checkpoint_path)
    else:
        print(f"检查点文件不存在: {checkpoint_path}")
        print("开始新训练")
        trainer.fit(raw_module, data_train, data_val)


@hydra.main(version_base="1.3", config_path="configs", config_name="rawdiffusion")
def main(cfg: DictConfig) -> None:
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description="从检查点恢复训练")
    parser.add_argument("--ckpt", type=str, help="检查点文件路径")
    parser.add_argument("--auto", action="store_true", help="自动检测最新检查点")
    
    # 解析Hydra配置后的剩余参数
    args = parser.parse_args()
    
    checkpoint_path = None
    if args.ckpt:
        checkpoint_path = args.ckpt
        print(f"使用指定检查点: {checkpoint_path}")
    elif args.auto:
        print("自动检测最新检查点")
        # checkpoint_path将在resume_training中自动检测
    
    resume_training(cfg, checkpoint_path)


if __name__ == "__main__":
    main() 