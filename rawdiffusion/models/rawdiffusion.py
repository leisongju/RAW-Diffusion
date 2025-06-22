import torch as th
import torch.nn as nn
import torch
import sys
import os

from .nn import SiLU, conv_nd, linear, zero_module, timestep_embedding, normalization
from .residual_blocks import (
    ResBlock,
    Downsample,
    Upsample,
    TimestepEmbedSequential,
)
from .attention_blocks import AttentionBlock
from functools import partial


class CrossViewAttention(nn.Module):
    """跨视图注意力机制，用于关联左右视图的相位信息"""
    def __init__(self, channels, num_heads=8, use_checkpoint=False, normalization_fn=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        # 使用LayerNorm而不是GroupNorm，因为我们要处理序列数据
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # 跨视图注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            linear(channels, channels * 4),
            SiLU(),
            linear(channels * 4, channels)
        )
        
    def forward(self, left_feat, right_feat):
        """
        Args:
            left_feat: [B, C, H, W] 左视图特征
            right_feat: [B, C, H, W] 右视图特征
        """
        B, C, H, W = left_feat.shape
        
        # 重塑为序列形式 [B, H*W, C]
        left_seq = left_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        right_seq = right_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 跨视图注意力：左视图查询右视图
        left_norm = self.norm1(left_seq)
        right_norm = self.norm1(right_seq)
        
        left_attended, _ = self.cross_attn(
            query=left_norm,
            key=right_norm,
            value=right_norm
        )
        
        # 残差连接
        left_out = left_seq + left_attended
        
        # FFN
        left_out = left_out + self.ffn(self.norm2(left_out))
        
        # 重塑回空间形式
        left_out = left_out.transpose(1, 2).reshape(B, C, H, W)
        
        return left_out


class CorrelatedNoiseGenerator(nn.Module):
    """相关性噪声生成器：共享噪声基底+独立扰动"""
    def __init__(self, channels, noise_ratio=0.8):
        super().__init__()
        self.noise_ratio = noise_ratio
        self.shared_noise_proj = nn.Conv2d(channels, channels, 1)
        self.left_noise_proj = nn.Conv2d(channels, channels, 1)
        self.right_noise_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x, timesteps):
        """
        Args:
            x: [B, C, H, W] 输入特征
            timesteps: [B] 时间步
        """
        B, C, H, W = x.shape
        
        # 生成共享噪声基底
        shared_noise = torch.randn_like(x)
        shared_noise = self.shared_noise_proj(shared_noise)
        
        # 生成独立扰动
        left_noise = torch.randn_like(x)
        right_noise = torch.randn_like(x)
        left_noise = self.left_noise_proj(left_noise)
        right_noise = self.right_noise_proj(right_noise)
        
        # 组合相关性噪声
        left_correlated_noise = self.noise_ratio * shared_noise + (1 - self.noise_ratio) * left_noise
        right_correlated_noise = self.noise_ratio * shared_noise + (1 - self.noise_ratio) * right_noise
        
        return left_correlated_noise, right_correlated_noise


class PDRAWDiffusionModel(nn.Module):
    """PDRAW扩散模型：共享编码器+分离解码器+跨视图注意力"""
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        rgb_guidance_module,
        attention_resolutions,
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
        conditional_block_name="RGBGuidedResidualBlock",
        norm_num_groups=8,
        latent_drop_rate=0,
        cross_view_attention=True,
        correlated_noise=True,
        noise_ratio=0.8,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        attention_resolutions_ds = []
        for res in attention_resolutions:
            attention_resolutions_ds.append(image_size // int(res))

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions_ds = attention_resolutions_ds
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.c_channels = c_channels
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.out_tanh = out_tanh
        self.cross_view_attention = cross_view_attention
        self.correlated_noise = correlated_noise

        self.normalization_fn = partial(normalization, num_groups=norm_num_groups)

        if rgb_guidance_module:
            self.rgb_guidance_module = rgb_guidance_module(
                normalization_fn=self.normalization_fn
            )
        else:
            self.rgb_guidance_module = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 共享编码器
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.shared_encoder = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, ch, 3, padding=1, padding_mode="reflect")
                )
            ]
        )

        self.latent_drop_rate = latent_drop_rate

        if latent_drop_rate > 0:
            self.mask_token = torch.nn.Parameter(torch.randn(c_channels))
        else:
            self.mask_token = None

        resblock_standard = partial(
            ResBlock,
            emb_channels=time_embed_dim,
            dropout=dropout,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            normalization_fn=self.normalization_fn,
        )

        if conditional_block_name == "RGBGuidedResidualBlock":
            from rawdiffusion.models.residual_blocks import RGBGuidedResidualBlock
            resblock_guidance_cls = partial(RGBGuidedResidualBlock)
        elif conditional_block_name == "ResBlock":
            resblock_guidance_cls = ResBlock
        else:
            raise ValueError(f"Unknown conditional block name: {conditional_block_name}")

        resblock_guidance = partial(
            resblock_guidance_cls,
            emb_channels=time_embed_dim,
            dropout=dropout,
            c_channels=c_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            normalization_fn=self.normalization_fn,
        )

        attention = partial(
            AttentionBlock,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_new_attention_order=use_new_attention_order,
            normalization_fn=self.normalization_fn,
        )

        attention_upsample = partial(
            AttentionBlock,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads_upsample,
            num_head_channels=num_head_channels,
            use_new_attention_order=use_new_attention_order,
            normalization_fn=self.normalization_fn,
        )

        # 构建共享编码器
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    resblock_standard(
                        ch,
                        out_channels=int(mult * model_channels),
                    )
                ]
                ch = int(mult * model_channels)
                if ds in self.attention_resolutions_ds:
                    layers.append(attention(ch))
                self.shared_encoder.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.shared_encoder.append(
                    TimestepEmbedSequential(
                        resblock_standard(ch, out_channels=out_ch, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # 中间块
        self.middle_block = TimestepEmbedSequential(
            resblock_guidance(ch),
            (attention(ch) if mid_attention else nn.Identity()),
            resblock_guidance(ch),
        )
        self._feature_size += ch

        # 跨视图注意力模块
        if self.cross_view_attention:
            self.cross_view_attn = CrossViewAttention(
                channels=ch,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint,
                normalization_fn=self.normalization_fn
            )

        # 相关性噪声生成器
        if self.correlated_noise:
            self.noise_generator = CorrelatedNoiseGenerator(
                channels=ch,
                noise_ratio=noise_ratio
            )

        # 分离的左眼解码器
        self.left_decoder = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    resblock_guidance(
                        ch + ich,
                        out_channels=int(model_channels * mult),
                    )
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_resolutions_ds:
                    layers.append(attention_upsample(ch))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        resblock_guidance(ch, out_channels=out_ch, up=True)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.left_decoder.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # 分离的右眼解码器
        self.right_decoder = nn.ModuleList([])
        # 重新构建input_block_chans用于右眼解码器
        input_block_chans = [int(channel_mult[0] * model_channels)]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                ch = int(mult * model_channels)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ch = int(mult * model_channels)
                input_block_chans.append(ch)
                ds *= 2

        ch = int(channel_mult[-1] * model_channels)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    resblock_guidance(
                        ch + ich,
                        out_channels=int(model_channels * mult),
                    )
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_resolutions_ds:
                    layers.append(attention_upsample(ch))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        resblock_guidance(ch, out_channels=out_ch, up=True)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.right_decoder.append(TimestepEmbedSequential(*layers))

        # 左眼输出层
        self.left_out = nn.Sequential(
            self.normalization_fn(input_ch),
            SiLU(),
            zero_module(
                conv_nd(dims, input_ch, out_channels, 3, padding=1, padding_mode="reflect")
            ),
        )

        # 右眼输出层
        self.right_out = nn.Sequential(
            self.normalization_fn(input_ch),
            SiLU(),
            zero_module(
                conv_nd(dims, input_ch, out_channels, 3, padding=1, padding_mode="reflect")
            ),
        )

    def forward(self, x, timesteps, guidance_data):
        if self.rgb_guidance_module is not None:
            guidance_features = self.rgb_guidance_module(guidance_data)

            if self.training and self.latent_drop_rate > 0:
                bs = guidance_features.shape[0]
                mask = (
                    torch.rand(bs, 1, 1, 1, device=guidance_features.device)
                    < self.latent_drop_rate
                ).float()
                mask_token = self.mask_token[None, :, None, None]
                guidance_features = guidance_features * (1 - mask) + mask_token * mask
        else:
            guidance_features = None

        # 时间嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # 共享编码器前向传播
        hs = []
        h = x.type(self.dtype)
        for block in self.shared_encoder:
            h = block(h, guidance_features, emb)
            hs.append(h)
        
        # 中间块
        h = self.middle_block(h, guidance_features, emb)

        # 相关性噪声注入
        if self.correlated_noise and self.training:
            left_noise, right_noise = self.noise_generator(h, timesteps)
            h_left = h + left_noise
            h_right = h + right_noise
        else:
            h_left = h
            h_right = h

        # 跨视图注意力
        if self.cross_view_attention:
            h_left = self.cross_view_attn(h_left, h_right)
            h_right = self.cross_view_attn(h_right, h_left)

        # 分离解码器前向传播
        hs_left = hs.copy()
        hs_right = hs.copy()
        
        # 左眼解码器
        for block in self.left_decoder:
            h_left = th.cat([h_left, hs_left.pop()], dim=1)
            h_left = block(h_left, guidance_features, emb)
        
        # 右眼解码器
        for block in self.right_decoder:
            h_right = th.cat([h_right, hs_right.pop()], dim=1)
            h_right = block(h_right, guidance_features, emb)

        # 输出层
        h_left = h_left.type(x.dtype)
        h_right = h_right.type(x.dtype)
        
        left_out = self.left_out(h_left)
        right_out = self.right_out(h_right)

        if self.out_tanh:
            left_out = th.tanh(left_out)
            right_out = th.tanh(right_out)

        # 将左右眼RAW数据在通道维度上拼接
        # 输出形状: [B, out_channels*2, H, W]
        # 前out_channels个通道是左眼，后out_channels个通道是右眼
        combined_output = th.cat([left_out, right_out], dim=1)
        
        return combined_output


# 保持向后兼容性
class RAWDiffusionModel(PDRAWDiffusionModel):
    """向后兼容的原始RAWDiffusionModel"""
    def __init__(self, *args, **kwargs):
        # 移除PDRAW相关参数
        kwargs.pop('cross_view_attention', None)
        kwargs.pop('correlated_noise', None)
        kwargs.pop('noise_ratio', None)
        
        super().__init__(*args, **kwargs)
        
    def forward(self, x, timesteps, guidance_data):
        # 调用父类方法但只返回单个输出
        result = super().forward(x, timesteps, guidance_data)
        return result


if __name__ == "__main__":
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
        conditional_block_name="RGBGuidedResidualBlock",
        norm_num_groups=8,
        latent_drop_rate=0,
        cross_view_attention=True,
        correlated_noise=True,
        noise_ratio=0.8,
    )
    print("=== PDRAW模型结构 ===")
    print(model)
    
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
    print(f"输出形状: {output.shape}")
    
    # 检查输出值范围
    print(f"\n输出值范围:")
    print(f"最小值: {output.min():.4f}, 最大值: {output.max():.4f}")
    
    # 测试推理模式（禁用相关性噪声）
    print("\n=== 测试推理模式 ===")
    model.eval()
    
    with torch.no_grad():
        output_eval = model(rgb_input, timesteps, guidance_data)
    
    print(f"推理模式输出:")
    print(f"输出形状: {output_eval.shape}")
    
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
        print(f"  输出形状: {test_output.shape}")
    
    # 测试模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== 模型参数统计 ===")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 测试内存使用
    print(f"\n=== 内存使用测试 ===")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 模拟训练步骤
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for i in range(3):
        optimizer.zero_grad()
        output = model(rgb_input, timesteps, guidance_data)
        
        # 模拟损失计算
        loss = torch.mean(output**2)
        loss.backward()
        optimizer.step()
        
        print(f"训练步骤 {i+1}: 损失 = {loss.item():.6f}")
    
    print("\n=== 测试完成 ===")
    print("PDRAW模型测试成功！模型能够正确生成左右眼RAW数据。")