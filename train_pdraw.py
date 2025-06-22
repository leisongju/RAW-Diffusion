import os
from typing import Any

import hydra
import lightning.pytorch as pl
import lightning.pytorch.callbacks as callbacks
import torch
import torch.optim.lr_scheduler as lr_scheduler
from aim.pytorch_lightning import AimLogger
from hydra.utils import instantiate
from lightning.pytorch.core import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torchinfo import summary
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from rawdiffusion.datasets.dataset_factory import create_dataset
from rawdiffusion.evaluation.collection import CollectionMetric
from rawdiffusion.evaluation.metrics import (
    MSEMetric,
    PearsonMetric,
    PSNRMetric,
    SSIMMetric,
)
from rawdiffusion.resample import create_named_schedule_sampler
from rawdiffusion.gaussian_diffusion_factory import (
    create_gaussian_diffusion,
)
from rawdiffusion.utils import get_output_path
from rawdiffusion.config import mod_config
from rawdiffusion.utils import rggb_to_rgb


class PDRAWDiffusionModule(LightningModule):
    def __init__(self, experiment_folder, **hparams) -> None:
        super().__init__()

        self.params = DictConfig(hparams)
        self.log_folder = experiment_folder
        self.save_hyperparameters()

        in_channels = self.params.model.in_channels
        image_size = self.params.general.image_size

        # 使用PDRAW模型，输入是RGB（3通道），输出是左右眼RAW拼接（6通道）
        self.model = instantiate(self.params.model, image_size=image_size, in_channels=6, out_channels=3)
        self.diffusion = create_gaussian_diffusion(**self.params.diffusion)
        self.diffusion_val = create_gaussian_diffusion(**self.params.diffusion_val)
        self.schedule_sampler = create_named_schedule_sampler(
            self.params.general.schedule_sampler, self.diffusion
        )

        summary(
            self.model,
            input_size=[
                (1, 6, image_size, image_size),
                (1,),
                (1, 3, image_size, image_size),
            ],
            depth=2,
        )

    def normalize_inv(self, x):
        return (x + 1) / 2.0

    def setup(self, stage: str) -> None:
        self.logger.experiment["hparams"] = self.params

    def forward_step(self, input_data, guidance_input, sampling_seed=None):
        t, weights = self.schedule_sampler.sample(
            input_data.shape[0], self.device, seed=sampling_seed
        )
        losses, extra = self.diffusion.training_losses(
            self.model,
            input_data,
            t,
            model_kwargs=guidance_input,
            weight_l2=self.params.general.weight_l2,
            weight_l1=self.params.general.weight_l1,
            weight_logl1=self.params.general.weight_logl1,
        )

        loss = (losses["loss"] * weights).mean()
        metrics = {k: v * weights for k, v in losses.items() if k != "loss"}

        return loss, extra, metrics

    def training_step(self, batch, batch_idx):
        # PDRAW数据：包含左右眼RAW
        left_raw_data = batch["left_raw_data"]
        right_raw_data = batch["right_raw_data"]
        guidance_data = batch["guidance_data"]

        guidance_input = self.preprocess_guidance(guidance_data)

        # 将左右眼RAW数据拼接成单个tensor
        # 形状: [B, out_channels*2, H, W]
        combined_raw_data = torch.cat([left_raw_data, right_raw_data], dim=1)

        # 计算损失
        loss, extra, metrics = self.forward_step(combined_raw_data, guidance_input)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for k, v in metrics.items():
            self.log(
                f"train_{k}",
                v.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        if (
            self.global_step % self.params.general.log_train_images_interval == 0
            and self.global_step >= 0
        ):
            self.log_batch_results(guidance_input, extra)
        if (
            self.global_step % self.params.general.log_train_images_interval == 0
            and self.global_step >= 0
        ):
            self.log_sampling_images(batch)

        return loss

    def on_validation_start(self) -> None:
        # 创建评估指标
        self.metrics_sampling = CollectionMetric(
            {
                "mse_rggb": MSEMetric(),
                "psnr_rggb": PSNRMetric(),
                "ssim_rggb": SSIMMetric(),
                "peason_rggb": PearsonMetric(),
                "mse_rgb": MSEMetric(rggb_to_rgb=True),
                "psnr_rgb": PSNRMetric(rggb_to_rgb=True),
                "ssim_rgb": SSIMMetric(rggb_to_rgb=True),
                "peason_rgb": PearsonMetric(rggb_to_rgb=True),
            }
        )

        self.eval_diffusion_process = (
            self.current_epoch + 1
        ) % self.params.general.eval_diffusion_process_interval == 0
        print("validation_start", self.current_epoch, self.eval_diffusion_process)

    def validation_step(self, batch, batch_idx):
        left_raw_data = batch["left_raw_data"]
        right_raw_data = batch["right_raw_data"]
        guidance_data = batch["guidance_data"]

        guidance_input = self.preprocess_guidance(guidance_data)

        sampling_seed = 123 + batch_idx
        
        # 将左右眼RAW数据拼接成单个tensor
        combined_raw_data = torch.cat([left_raw_data, right_raw_data], dim=1)
        
        loss, extra, metrics = self.forward_step(
            combined_raw_data, guidance_input, sampling_seed=sampling_seed
        )

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        val_steps_per_epoch = self.trainer.num_val_batches[0]
        sampling_interval = (
            val_steps_per_epoch // self.params.general.val_sampling_frequency
        )

        sampling = batch_idx % sampling_interval == 0
        sampling_log_images = (
            batch_idx // sampling_interval
        ) % self.params.general.log_val_sampling_images_interval == 0

        if sampling and sampling_log_images:
            filename = f"{batch_idx:04d}_{(self.global_step):06d}.png"
            self.log_batch_results(
                guidance_input, extra, mode="validation", filename=filename
            )

        if self.eval_diffusion_process and sampling:
            filename = f"{batch_idx:04d}_{(self.global_step):06d}.png"
            save_results = sampling_log_images
            raw_generated = self.log_sampling_images(
                batch,
                mode_name="validation_sampling",
                filename=filename,
                sampling_seed=sampling_seed,
                save_results=save_results,
            )
            
            # 将生成的拼接tensor分割回左右眼
            out_channels = self.params.model.out_channels
            left_raw_generated = raw_generated[:, :out_channels, :, :]
            right_raw_generated = raw_generated[:, out_channels:, :, :]
            
            # 分别评估左右眼
            left_raw_data_device = left_raw_data.to(left_raw_generated.device)
            right_raw_data_device = right_raw_data.to(right_raw_generated.device)

            # 计算左眼指标
            self.metrics_sampling.update(
                self.normalize_inv(left_raw_data_device), self.normalize_inv(left_raw_generated)
            )
            # 计算右眼指标
            self.metrics_sampling.update(
                self.normalize_inv(right_raw_data_device), self.normalize_inv(right_raw_generated)
            )

    def on_validation_epoch_end(self) -> None:
        if self.eval_diffusion_process:
            for k, v in self.metrics_sampling.compute().items():
                self.log(f"val_sampling_{k}", v)

    def preprocess_guidance(self, guidance_data):
        guidance_input = {}

        drop_rate = self.params.general.drop_rate
        bs = guidance_data.shape[0]
        if self.training and drop_rate > 0.0:
            mask = (
                (torch.rand([bs, 1, 1, 1]) > drop_rate).float().to(guidance_data.device)
            )
            guidance_data = guidance_data * mask

        guidance_input["guidance_data"] = guidance_data

        return guidance_input

    def log_batch_results(
        self, model_kwargs, return_dict, mode="training", filename=None
    ):
        x_start = return_dict["x_start"]
        x_t = return_dict["x_t"]
        model_output = return_dict["model_output"]
        target = return_dict["target"]
        guidance_data = model_kwargs["guidance_data"]
        
        # 处理拼接的tensor，分割为左右眼
        out_channels = self.params.model.out_channels
        
        if x_start.shape[1] == out_channels * 2:  # 如果是拼接的PDRAW数据
            # 分割左右眼数据
            x_start_left = x_start[:, :out_channels, :, :]
            x_start_right = x_start[:, out_channels:, :, :]
            x_t_left = x_t[:, :out_channels, :, :]
            x_t_right = x_t[:, out_channels:, :, :]
            model_output_left = model_output[:, :out_channels, :, :]
            model_output_right = model_output[:, out_channels:, :, :]
            target_left = target[:, :out_channels, :, :]
            target_right = target[:, out_channels:, :, :]
            
            # 转换为RGB进行可视化
            if out_channels == 4:  # RGGB格式
                x_start_left = self.rggb_to_rgb_and_gc(x_start_left)
                x_start_right = self.rggb_to_rgb_and_gc(x_start_right)
                x_t_left = self.rggb_to_rgb_and_gc(x_t_left)
                x_t_right = self.rggb_to_rgb_and_gc(x_t_right)
                model_output_left = self.rggb_to_rgb_and_gc(model_output_left)
                model_output_right = self.rggb_to_rgb_and_gc(model_output_right)
                target_left = self.rggb_to_rgb_and_gc(target_left)
                target_right = self.rggb_to_rgb_and_gc(target_right)
            
            # 创建左眼可视化
            vis_left = torch.concatenate(
                [
                    x_start_left,
                    guidance_data.to(x_start_left.device),
                    x_t_left,
                    model_output_left,
                    target_left,
                    model_output_left - target_left,
                ],
                dim=3,
            )
            
            # 创建右眼可视化
            vis_right = torch.concatenate(
                [
                    x_start_right,
                    guidance_data.to(x_start_right.device),
                    x_t_right,
                    model_output_right,
                    target_right,
                    model_output_right - target_right,
                ],
                dim=3,
            )
            
            vis_left = torch.clamp(vis_left, -1, 1)
            vis_right = torch.clamp(vis_right, -1, 1)
            vis_left = make_grid(vis_left, nrow=1)
            vis_right = make_grid(vis_right, nrow=1)
            vis_left = self.normalize_inv(vis_left)
            vis_right = self.normalize_inv(vis_right)

            if filename is None:
                filename = f"{(self.global_step):06d}.png"

            # 保存左眼结果
            vis_path_left = os.path.join(self.log_folder, mode, filename.replace('.png', '_left.png'))
            os.makedirs(os.path.dirname(vis_path_left), exist_ok=True)
            to_pil_image(vis_left).save(vis_path_left)

            # 保存右眼结果
            vis_path_right = os.path.join(self.log_folder, mode, filename.replace('.png', '_right.png'))
            os.makedirs(os.path.dirname(vis_path_right), exist_ok=True)
            to_pil_image(vis_right).save(vis_path_right)
            
        else:  # 原始单眼数据格式
            if x_start.shape[1] == 4:
                x_start = self.rggb_to_rgb_and_gc(x_start)
                x_t = self.rggb_to_rgb_and_gc(x_t)
                model_output = self.rggb_to_rgb_and_gc(model_output)
                target = self.rggb_to_rgb_and_gc(target)

            vis = torch.concatenate(
                [
                    x_start,
                    guidance_data.to(x_start.device),
                    x_t,
                    model_output,
                    target,
                    model_output - target,
                ],
                dim=3,
            )
            vis = torch.clamp(vis, -1, 1)
            vis = make_grid(vis, nrow=1)
            vis = self.normalize_inv(vis)

            if filename is None:
                filename = f"{(self.global_step):06d}.png"

            vis_path = os.path.join(self.log_folder, mode, filename)
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            to_pil_image(vis).save(vis_path)

    def log_sampling_images(
        self,
        batch,
        mode_name="training_sampling",
        filename=None,
        sampling_seed=None,
        save_results=True,
    ):
        left_raw_data = batch["left_raw_data"]
        right_raw_data = batch["right_raw_data"]
        guidance_data = batch["guidance_data"]

        guidance_input = self.preprocess_guidance(guidance_data)

        bs, _, h, w = left_raw_data.shape

        use_ddim = True
        clip_denoised = True
        diffusion = self.diffusion_val

        g = torch.Generator(device=self.device)
        if sampling_seed is not None:
            g.manual_seed(sampling_seed)

        with torch.inference_mode():
            # 注意：输入通道数需要是out_channels*2，因为我们要生成左右眼
            shape = (bs, self.params.model.out_channels * 2, h, w)
            noise = torch.randn(*shape, device=self.device, generator=g)

            sample_fn = (
                diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                model_kwargs=guidance_input,
                progress=True,
            )

        d = sample.device

        # 将生成的拼接tensor分割回左右眼
        out_channels = self.params.model.out_channels
        left_raw_generated = sample[:, :out_channels, :, :]
        right_raw_generated = sample[:, out_channels:, :, :]

        if save_results:
            # 处理左眼可视化
            if left_raw_data.shape[1] == 4:
                left_raw_data_rgb = self.rggb_to_rgb_and_gc(left_raw_data)
                left_raw_generated_rgb = self.rggb_to_rgb_and_gc(left_raw_generated)
            else:
                left_raw_data_rgb = left_raw_data
                left_raw_generated_rgb = left_raw_generated

            # 处理右眼可视化
            if right_raw_data.shape[1] == 4:
                right_raw_data_rgb = self.rggb_to_rgb_and_gc(right_raw_data)
                right_raw_generated_rgb = self.rggb_to_rgb_and_gc(right_raw_generated)
            else:
                right_raw_data_rgb = right_raw_data
                right_raw_generated_rgb = right_raw_generated

            # 左眼可视化
            vis_left = torch.concat([left_raw_data_rgb.to(d), guidance_data.to(d), left_raw_generated_rgb], dim=3)
            vis_left = self.normalize_inv(vis_left)
            vis_left = make_grid(vis_left, nrow=1)

            # 右眼可视化
            vis_right = torch.concat([right_raw_data_rgb.to(d), guidance_data.to(d), right_raw_generated_rgb], dim=3)
            vis_right = self.normalize_inv(vis_right)
            vis_right = make_grid(vis_right, nrow=1)

            if filename is None:
                filename = f"{(self.global_step):06d}.png"

            # 保存左眼结果
            vis_path_left = os.path.join(self.log_folder, mode_name, filename.replace('.png', '_left.png'))
            os.makedirs(os.path.dirname(vis_path_left), exist_ok=True)
            to_pil_image(vis_left).save(vis_path_left)

            # 保存右眼结果
            vis_path_right = os.path.join(self.log_folder, mode_name, filename.replace('.png', '_right.png'))
            os.makedirs(os.path.dirname(vis_path_right), exist_ok=True)
            to_pil_image(vis_right).save(vis_path_right)

        return sample

    @staticmethod
    def rggb_to_rgb_and_gc(data, gamma=1.0 / 5):
        data = rggb_to_rgb(data)

        data = (data + 1) / 2.0
        data = data**gamma
        data = data * 2 - 1

        return data

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(
            self.parameters(),
            lr=self.params.general.lr,
            weight_decay=self.params.general.weight_decay,
        )

        if self.params.general.lr_scheduler == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.params.general.max_steps,
            )
        elif self.params.general.lr_scheduler == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.params.general.max_steps, eta_min=0.0
            )
        else:
            raise ValueError(
                f"Unknown lr_scheduler: {self.params.general.lr_scheduler}"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


@hydra.main(version_base="1.3", config_path="configs", config_name="rawdiffusion")
def main(cfg: DictConfig) -> None:
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
        checkpoint_path = os.path.join(
            experiment_folder,
            "checkpoints",
        )
        checkpoint_cb = callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
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

    # 检查是否有检查点需要恢复
    checkpoint_to_resume = None
    
    # 方法1: 检查命令行参数中是否有ckpt_path
    if hasattr(cfg, 'ckpt_path') and cfg.ckpt_path:
        checkpoint_to_resume = cfg.ckpt_path
        print(f"从指定检查点恢复训练: {checkpoint_to_resume}")
    
    # 方法2: 自动检测last.ckpt
    elif cfg.general.checkpoint:
        last_ckpt_path = os.path.join(checkpoint_path, "last.ckpt")
        if os.path.exists(last_ckpt_path):
            checkpoint_to_resume = last_ckpt_path
            print(f"自动检测到last.ckpt，从检查点恢复训练: {checkpoint_to_resume}")
    
    # 方法3: 检查是否有其他检查点文件
    elif cfg.general.checkpoint:
        checkpoint_dir = os.path.join(experiment_folder, "checkpoints")
        if os.path.exists(checkpoint_dir):
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if ckpt_files:
                # 按修改时间排序，选择最新的
                ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                checkpoint_to_resume = os.path.join(checkpoint_dir, ckpt_files[0])
                print(f"检测到检查点文件，从最新检查点恢复训练: {checkpoint_to_resume}")

    if checkpoint_to_resume:
        print(f"正在从检查点恢复训练: {checkpoint_to_resume}")
        trainer.fit(raw_module, data_train, data_val, ckpt_path=checkpoint_to_resume)
    else:
        print("开始新的训练")
        trainer.fit(raw_module, data_train, data_val)


if __name__ == "__main__":
    main()
