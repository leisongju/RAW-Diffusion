# 从检查点恢复训练

本文档说明如何从 `last.ckpt` 或其他检查点文件恢复训练。

## 方法一：使用修改后的训练脚本（推荐）

原始的 `train_pdraw.py` 已经修改，现在会自动检测检查点：

```bash
# 自动检测last.ckpt并恢复训练
python train_pdraw.py

# 或者指定检查点路径
python train_pdraw.py ckpt_path=/path/to/your/checkpoint.ckpt
```

## 方法二：使用专门的恢复脚本

使用 `resume_training.py` 脚本：

```bash
# 自动检测最新检查点
python resume_training.py --auto

# 指定检查点路径
python resume_training.py --ckpt /path/to/your/checkpoint.ckpt

# 不指定参数，自动检测last.ckpt
python resume_training.py
```

## 方法三：直接使用PyTorch Lightning

```bash
# 使用last.ckpt
python train_pdraw.py trainer.resume_from_checkpoint=last.ckpt

# 指定完整路径
python train_pdraw.py trainer.resume_from_checkpoint=/full/path/to/checkpoint.ckpt
```

## 检查点文件位置

检查点文件通常保存在：
```
{experiment_folder}/checkpoints/
├── last.ckpt          # 最新的检查点
├── epoch=0-step=1000.ckpt
├── epoch=0-step=2000.ckpt
└── ...
```

## 注意事项

1. **确保配置一致**：恢复训练时使用的配置应该与保存检查点时一致
2. **数据路径**：确保数据集路径仍然有效
3. **依赖版本**：确保PyTorch、Lightning等依赖版本兼容
4. **GPU内存**：确保有足够的GPU内存加载模型

## 常见问题

### Q: 找不到检查点文件？
A: 检查 `experiment_folder/checkpoints/` 目录是否存在，以及是否有 `.ckpt` 文件。

### Q: 恢复训练后损失不连续？
A: 这是正常现象，因为优化器状态和学习率调度器会从检查点恢复。

### Q: 如何查看检查点信息？
```python
import torch

# 加载检查点
checkpoint = torch.load('path/to/checkpoint.ckpt', map_location='cpu')

# 查看检查点内容
print(checkpoint.keys())
print(f"Global step: {checkpoint['global_step']}")
print(f"Epoch: {checkpoint['epoch']}")
```

### Q: 如何修改训练步数？
在配置文件中修改 `general.max_steps`，或者在命令行中覆盖：
```bash
python train_pdraw.py general.max_steps=100000
``` 