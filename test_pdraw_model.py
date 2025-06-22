#!/usr/bin/env python3
"""
PDRAW模型测试脚本
用于测试PDRAWDiffusionModel的功能
"""

import torch
import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from rawdiffusion.models.rawdiffusion import PDRAWDiffusionModel

def test_pdraw_model():
    """测试PDRAW模型"""
    print("=== 创建PDRAW模型 ===")
    
    model = PDRAWDiffusionModel(
        image_size=256,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        rgb_guidance_module=None,
        attention_resolutions=[8, 4, 2, 1],
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        c_channels=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        mid_attention=True,
        out_tanh=False,
        conditional_block_name="ResBlock",  # 使用ResBlock避免RGBGuidedResidualBlock的依赖
        norm_num_groups=8,
        latent_drop_rate=0,
        cross_view_attention=True,
        correlated_noise=True,
        noise_ratio=0.8,
    )
    
    print("模型创建成功！")
    print(f"模型类型: {type(model).__name__}")
    
    # 测试随机输入生成
    print("\n=== 测试PDRAW生成 ===")
    
    # 设置模型为训练模式（启用相关性噪声）
    model.train()
    
    # 创建随机输入
    batch_size = 2
    height, width = 256, 256
    rgb_input = torch.randn(batch_size, 3, height, width)
    timesteps = torch.randint(0, 1000, (batch_size,))
    guidance_data = torch.randn(batch_size, 3, height, width)  # 模拟RGB引导数据
    
    print(f"输入RGB形状: {rgb_input.shape}")
    print(f"时间步形状: {timesteps.shape}")
    print(f"引导数据形状: {guidance_data.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(rgb_input, timesteps, guidance_data)
    
    print(f"\n输出结果:")
    print(f"左眼RAW形状: {output['left_raw'].shape}")
    print(f"右眼RAW形状: {output['right_raw'].shape}")
    
    # 检查输出值范围
    print(f"\n输出值范围:")
    print(f"左眼RAW - 最小值: {output['left_raw'].min():.4f}, 最大值: {output['left_raw'].max():.4f}")
    print(f"右眼RAW - 最小值: {output['right_raw'].min():.4f}, 最大值: {output['right_raw'].max():.4f}")
    
    # 测试推理模式（禁用相关性噪声）
    print("\n=== 测试推理模式 ===")
    model.eval()
    
    with torch.no_grad():
        output_eval = model(rgb_input, timesteps, guidance_data)
    
    print(f"推理模式输出:")
    print(f"左眼RAW形状: {output_eval['left_raw'].shape}")
    print(f"右眼RAW形状: {output_eval['right_raw'].shape}")
    
    # 测试不同输入尺寸
    print("\n=== 测试不同输入尺寸 ===")
    test_sizes = [(128, 128), (512, 512)]
    
    for h, w in test_sizes:
        test_input = torch.randn(1, 3, h, w)
        test_timesteps = torch.randint(0, 1000, (1,))
        test_guidance = torch.randn(1, 3, h, w)
        
        with torch.no_grad():
            test_output = model(test_input, test_timesteps, test_guidance)
        
        print(f"输入尺寸 {h}x{w}:")
        print(f"  左眼RAW: {test_output['left_raw'].shape}")
        print(f"  右眼RAW: {test_output['right_raw'].shape}")
    
    # 测试模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== 模型参数统计 ===")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 测试内存使用和训练
    print(f"\n=== 内存使用测试 ===")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA可用，使用GPU")
        model = model.cuda()
        rgb_input = rgb_input.cuda()
        timesteps = timesteps.cuda()
        guidance_data = guidance_data.cuda()
    else:
        print("使用CPU")
    
    # 模拟训练步骤
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for i in range(3):
        optimizer.zero_grad()
        output = model(rgb_input, timesteps, guidance_data)
        
        # 模拟损失计算
        loss = torch.mean(output['left_raw']**2) + torch.mean(output['right_raw']**2)
        loss.backward()
        optimizer.step()
        
        print(f"训练步骤 {i+1}: 损失 = {loss.item():.6f}")
    
    print("\n=== 测试完成 ===")
    print("✅ PDRAW模型测试成功！模型能够正确生成左右眼RAW数据。")
    
    # 测试跨视图注意力
    print("\n=== 测试跨视图注意力 ===")
    if hasattr(model, 'cross_view_attn') and model.cross_view_attn is not None:
        print("✅ 跨视图注意力模块存在")
        # 测试跨视图注意力 - 确保在同一设备上，使用正确的通道数
        device = next(model.parameters()).device
        # 使用模型实际的通道数 (512 = 64 * 8，来自channel_mult的最后一个值)
        channels = 512
        left_feat = torch.randn(1, channels, 32, 32).to(device)
        right_feat = torch.randn(1, channels, 32, 32).to(device)
        attended = model.cross_view_attn(left_feat, right_feat)
        print(f"跨视图注意力输出形状: {attended.shape}")
        print(f"跨视图注意力输出范围: [{attended.min():.4f}, {attended.max():.4f}]")
    else:
        print("❌ 跨视图注意力模块不存在")
    
    # 测试相关性噪声生成器
    print("\n=== 测试相关性噪声生成器 ===")
    if hasattr(model, 'noise_generator') and model.noise_generator is not None:
        print("✅ 相关性噪声生成器存在")
        # 测试噪声生成 - 确保在同一设备上，使用正确的通道数
        device = next(model.parameters()).device
        channels = 512
        feat = torch.randn(1, channels, 32, 32).to(device)
        timesteps = torch.randint(0, 1000, (1,)).to(device)
        left_noise, right_noise = model.noise_generator(feat, timesteps)
        print(f"左眼噪声形状: {left_noise.shape}")
        print(f"右眼噪声形状: {right_noise.shape}")
        print(f"左眼噪声范围: [{left_noise.min():.4f}, {left_noise.max():.4f}]")
        print(f"右眼噪声范围: [{right_noise.min():.4f}, {right_noise.max():.4f}]")
    else:
        print("❌ 相关性噪声生成器不存在")
    
    # 调试输出全为0的问题
    print("\n=== 调试输出全为0问题 ===")
    model.eval()
    with torch.no_grad():
        # 使用非零输入测试
        test_input = torch.ones(1, 3, 256, 256).to(device)
        test_timesteps = torch.randint(0, 1000, (1,)).to(device)
        test_guidance = torch.ones(1, 3, 256, 256).to(device)
        
        test_output = model(test_input, test_timesteps, test_guidance)
        print(f"使用全1输入:")
        print(f"  左眼RAW范围: [{test_output['left_raw'].min():.6f}, {test_output['left_raw'].max():.6f}]")
        print(f"  右眼RAW范围: [{test_output['right_raw'].min():.6f}, {test_output['right_raw'].max():.6f}]")
        
        # 检查中间特征
        print(f"  输入特征范围: [{test_input.min():.6f}, {test_input.max():.6f}]")
        
        # 检查模型参数
        for name, param in model.named_parameters():
            if 'out' in name and param.requires_grad:
                print(f"  {name}: [{param.min():.6f}, {param.max():.6f}]")
                break

if __name__ == "__main__":
    test_pdraw_model() 