#!/usr/bin/env python3
"""
检查检查点文件信息的脚本
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from datetime import datetime

def check_checkpoint(checkpoint_path):
    """检查检查点文件信息"""
    if not os.path.exists(checkpoint_path):
        print(f"错误：检查点文件不存在: {checkpoint_path}")
        return
    
    print(f"检查点文件: {checkpoint_path}")
    print(f"文件大小: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
    print(f"修改时间: {datetime.fromtimestamp(os.path.getmtime(checkpoint_path))}")
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("\n检查点内容:")
        print(f"Keys: {list(checkpoint.keys())}")
        
        if 'global_step' in checkpoint:
            print(f"Global step: {checkpoint['global_step']}")
        
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"模型参数数量: {len(state_dict)}")
            
            # 计算参数总数
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"总参数数量: {total_params:,}")
        
        if 'optimizer_states' in checkpoint:
            print(f"优化器状态: 已保存")
        
        if 'lr_schedulers' in checkpoint:
            print(f"学习率调度器: 已保存")
        
        if 'callbacks' in checkpoint:
            print(f"回调函数: 已保存")
            
    except Exception as e:
        print(f"加载检查点时出错: {e}")

def find_checkpoints(experiment_folder):
    """查找实验文件夹中的所有检查点"""
    checkpoint_dir = os.path.join(experiment_folder, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print(f"检查点目录不存在: {checkpoint_dir}")
        return []
    
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not ckpt_files:
        print(f"未找到检查点文件在: {checkpoint_dir}")
        return []
    
    # 按修改时间排序
    ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    print(f"找到 {len(ckpt_files)} 个检查点文件:")
    for i, ckpt_file in enumerate(ckpt_files):
        full_path = os.path.join(checkpoint_dir, ckpt_file)
        mtime = datetime.fromtimestamp(os.path.getmtime(full_path))
        size = os.path.getsize(full_path) / (1024*1024)
        print(f"  {i+1}. {ckpt_file} ({size:.2f} MB, {mtime})")
    
    return [os.path.join(checkpoint_dir, f) for f in ckpt_files]

def main():
    parser = argparse.ArgumentParser(description="检查检查点文件信息")
    parser.add_argument("--ckpt", type=str, help="检查点文件路径")
    parser.add_argument("--exp", type=str, help="实验文件夹路径")
    parser.add_argument("--list", action="store_true", help="列出所有检查点")
    
    args = parser.parse_args()
    
    if args.ckpt:
        # 检查指定检查点
        check_checkpoint(args.ckpt)
    elif args.exp:
        # 查找实验文件夹中的检查点
        checkpoints = find_checkpoints(args.exp)
        if checkpoints and args.list:
            print(f"\n检查最新的检查点:")
            check_checkpoint(checkpoints[0])
    else:
        print("请指定检查点文件路径 (--ckpt) 或实验文件夹路径 (--exp)")
        print("使用 --list 选项来检查最新的检查点")

if __name__ == "__main__":
    main() 