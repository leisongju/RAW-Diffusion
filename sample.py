import os

import h5py
import hydra
import numpy as np
import torch
import torchvision as tv
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from rawdiffusion.evaluation.collection import CollectionMetric
from rawdiffusion.evaluation.metrics import (
    MSEMetric,
    PearsonMetric,
    PSNRMetric,
    SSIMMetric,
)
from rawdiffusion.datasets.dataset_factory import create_dataset
from rawdiffusion.gaussian_diffusion_factory import create_gaussian_diffusion
from rawdiffusion.utils import get_output_path
from train_pdraw import PDRAWDiffusionModule
from rawdiffusion.config import mod_config
from rawdiffusion.utils import create_folder_for_file
from rawdiffusion.utils import gamma_correction, rggb_to_rgb


def get_val_output_name(cfg):
    output_name = cfg.dataset.val.data_dir
    output_name = os.path.basename(os.path.normpath(output_name))

    file_list_name = os.path.splitext(os.path.basename(cfg.dataset.val.file_list))[0]
    output_name += "_" + file_list_name

    if cfg.diffusion_val.timestep_respacing:
        output_name += "_" + cfg.diffusion_val.timestep_respacing

    return output_name


@hydra.main(
    version_base="1.3", config_path="configs", config_name="rawdiffusion_sample"
)
def main(cfg: DictConfig) -> None:
    mod_config(cfg)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    experiment_folder = get_output_path(cfg)
    print(f"experiment_folder: {experiment_folder}")

    checkpoint_path = os.path.join(
        experiment_folder, "checkpoints", cfg.checkpoint_name
    )
    print(f"checkpoint_path: {checkpoint_path}")

    output_name = get_val_output_name(cfg)
    print(f"inference output name: {output_name}")

    raw_module = PDRAWDiffusionModule.load_from_checkpoint(
        checkpoint_path, experiment_folder=experiment_folder, **cfg
    )

    data_val = create_dataset(
        **cfg.dataset.val,
        transform=False,
        seed=cfg.general.seed,
    )
    raw_module.eval()

    image_path = os.path.join(
        experiment_folder, "inference_sampling", output_name, "visualizations"
    )
    os.makedirs(image_path, exist_ok=True)
    inference_output_path = os.path.join(
        experiment_folder,
        "inference_sampling",
        output_name,
    )
    os.makedirs(inference_output_path, exist_ok=True)

    config_path = os.path.join(
        experiment_folder, "inference_sampling", output_name, "config.yaml"
    )
    metric_path = os.path.join(
        experiment_folder, "inference_sampling", output_name, "metric.yaml"
    )

    metrics_sampling = CollectionMetric(
        {
            # "mse_rggb": MSEMetric(),
            # "psnr_rggb": PSNRMetric(),
            # "ssim_rggb": SSIMMetric(),
            # "pearson_rggb": PearsonMetric(),
            "mse_rgb": MSEMetric(rggb_to_rgb=True),
            "psnr_rgb": PSNRMetric(rggb_to_rgb=True),
            "ssim_rgb": SSIMMetric(rggb_to_rgb=True),
            "pearson_rgb": PearsonMetric(rggb_to_rgb=True),
        }
    )

    diffusion = create_gaussian_diffusion(**cfg.diffusion_val)

    use_ddim = "ddim" in cfg.diffusion_val.timestep_respacing
    clip_denoised = True

    save_visualization_interval = cfg.save_visualization_interval
    save_timesteps = cfg.save_timesteps
    save_pred = cfg.save_pred
    save_tar = cfg.save_tar
    rgb_only = cfg.rgb_only

    save_as_hdf5 = cfg.save_as_hdf5

    model = raw_module.model

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("PSNR: {task.fields[psnr_rgb]}"),
        TextColumn("SSIM: {task.fields[ssim_rgb]}"),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task_total_id = progress.add_task(
            "[red]Total Dataset", total=len(data_val), psnr_rgb="", ssim_rgb=""
        )
        task_batch_id = progress.add_task(
            "[green]Batch", total=100, psnr_rgb="", ssim_rgb=""
        )
        task_total = progress._tasks[task_total_id]

        print("sampling...")
        num_samples = 0

        for batch in data_val:
            # PDRAW数据：包含左右眼RAW
            left_raw_data = batch["left_raw_data"].cuda()
            right_raw_data = batch["right_raw_data"].cuda()
            guidance_data = batch["guidance_data"].cuda()

            guidance_input = raw_module.preprocess_guidance(guidance_data)
            guidance_input = {k: v.cuda() for k, v in guidance_input.items()}

            progress.reset(task_batch_id, total=diffusion.num_timesteps)

            # 将左右眼RAW数据拼接成单个tensor
            combined_raw_data = torch.cat([left_raw_data, right_raw_data], dim=1)
            
            ts = diffusion.num_timesteps - 1
            noise = torch.randn_like(combined_raw_data)

            indices = list(range(ts))[::-1]
            sample_fn_progressive = (
                diffusion.p_sample_loop_progressive
                if not use_ddim
                else diffusion.ddim_sample_loop_progressive
            )

            vis_step = max(1, diffusion.num_timesteps // 8)
            samples = []

            with torch.inference_mode():
                for sample_dict in sample_fn_progressive(
                    model,
                    shape=(
                        guidance_data.shape[0],
                        6,  # 左右眼拼接：3+3=6通道
                        guidance_data.shape[2],
                        guidance_data.shape[3],
                    ),
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=guidance_input,
                    device=None,
                    progress=False,
                    progress_fn=lambda: progress.advance(task_batch_id, advance=1),
                    indices=indices,
                ):
                    sample_t = sample_dict["t"]
                    sample = sample_dict["sample"]
                    if sample_t % vis_step == 0 or sample_t == 0:
                        samples.append(sample)

            sample = (sample + 1) / 2.0
            combined_raw_data = (combined_raw_data + 1) / 2.0
            samples = [(sample + 1) / 2.0 for sample in samples]
            noise = (noise + 1) / 2.0
            guidance_data = (guidance_data + 1) / 2.0

            # 将生成的拼接tensor分割回左右眼
            left_raw_generated = sample[:, :3, :, :]  # 前3通道是左眼
            right_raw_generated = sample[:, 3:, :, :]  # 后3通道是右眼
            left_raw_data_separated = combined_raw_data[:, :3, :, :]  # 前3通道是左眼
            right_raw_data_separated = combined_raw_data[:, 3:, :, :]  # 后3通道是右眼

            if not rgb_only:
                # 分别评估左右眼
                metrics_sampling.update(left_raw_data_separated, left_raw_generated)
                metrics_sampling.update(right_raw_data_separated, right_raw_generated)

            # 转换为RGB进行可视化
            if left_raw_data_separated.shape[1] == 4:  # RGGB格式
                left_batch_rgb = rggb_to_rgb(left_raw_data_separated)
                right_batch_rgb = rggb_to_rgb(right_raw_data_separated)
                left_sample_rgb = rggb_to_rgb(left_raw_generated)
                right_sample_rgb = rggb_to_rgb(right_raw_generated)
                samples_left_rgb = [rggb_to_rgb(s[:, :3, :, :]) for s in samples]
                samples_right_rgb = [rggb_to_rgb(s[:, 3:, :, :]) for s in samples]
            else:
                left_batch_rgb = left_raw_data_separated
                right_batch_rgb = right_raw_data_separated
                left_sample_rgb = left_raw_generated
                right_sample_rgb = right_raw_generated
                samples_left_rgb = [s[:, :3, :, :] for s in samples]
                samples_right_rgb = [s[:, 3:, :, :] for s in samples]

            left_batch_rgb_gc = gamma_correction(left_batch_rgb)
            right_batch_rgb_gc = gamma_correction(right_batch_rgb)
            left_sample_rgb_gc = gamma_correction(left_sample_rgb)
            right_sample_rgb_gc = gamma_correction(right_sample_rgb)
            samples_left_rgb_gc = [gamma_correction(s) for s in samples_left_rgb]
            samples_right_rgb_gc = [gamma_correction(s) for s in samples_right_rgb]

            left_sample_np = left_raw_generated.cpu().numpy()
            right_sample_np = right_raw_generated.cpu().numpy()
            left_batch_np = left_raw_data_separated.cpu().numpy()
            right_batch_np = right_raw_data_separated.cpu().numpy()

            for j in range(sample.shape[0]):
                rel_path = batch["path"][j]

                if os.path.isabs(rel_path):
                    rel_path = os.path.basename(rel_path)
                fn = os.path.splitext(rel_path)[0]

                if torch.isnan(left_sample_rgb[j]).any() or torch.isnan(right_sample_rgb[j]).any():
                    print("sample is nan. skipping")
                    continue

                if num_samples % save_visualization_interval == 0:
                    gt_data_path = os.path.join(image_path, fn + "_gt.png")
                    create_folder_for_file(gt_data_path)
                    if not rgb_only:
                        tv.utils.save_image(left_batch_rgb_gc[j], os.path.join(image_path, fn + "_left_gt.png"))
                        tv.utils.save_image(right_batch_rgb_gc[j], os.path.join(image_path, fn + "_right_gt.png"))
                    tv.utils.save_image(left_sample_rgb_gc[j], os.path.join(image_path, fn + "_left.png"))
                    tv.utils.save_image(right_sample_rgb_gc[j], os.path.join(image_path, fn + "_right.png"))
                    tv.utils.save_image(
                        guidance_data[j], os.path.join(image_path, fn + "_rgb.png")
                    )

                    if save_timesteps:
                        for t, (s_left, s_right) in enumerate(zip(samples_left_rgb_gc, samples_right_rgb_gc)):
                            tv.utils.save_image(
                                s_left[j], os.path.join(image_path, f"{fn}_left_{t}.png")
                            )
                            tv.utils.save_image(
                                s_right[j], os.path.join(image_path, f"{fn}_right_{t}.png")
                            )

                if save_pred:
                    if save_as_hdf5:
                        left_pred_np_path = os.path.join(
                            inference_output_path, fn + "_left_pred_u16.hdf5"
                        )
                        create_folder_for_file(left_pred_np_path)
                        left_sample_data = left_sample_np[j].transpose(1, 2, 0)
                        left_sample_data = (left_sample_data * 65535).astype(np.uint16)
                        with h5py.File(left_pred_np_path, "w") as f:
                            f.create_dataset(
                                "raw",
                                data=left_sample_data,
                                compression="gzip",
                                compression_opts=9,
                            )

                        right_pred_np_path = os.path.join(
                            inference_output_path, fn + "_right_pred_u16.hdf5"
                        )
                        create_folder_for_file(right_pred_np_path)
                        right_sample_data = right_sample_np[j].transpose(1, 2, 0)
                        right_sample_data = (right_sample_data * 65535).astype(np.uint16)
                        with h5py.File(right_pred_np_path, "w") as f:
                            f.create_dataset(
                                "raw",
                                data=right_sample_data,
                                compression="gzip",
                                compression_opts=9,
                            )
                    else:
                        left_pred_np_path = os.path.join(
                            inference_output_path, fn + "_left_pred.npy"
                        )
                        create_folder_for_file(left_pred_np_path)
                        np.save(left_pred_np_path, left_sample_np[j].transpose(1, 2, 0))

                        right_pred_np_path = os.path.join(
                            inference_output_path, fn + "_right_pred.npy"
                        )
                        create_folder_for_file(right_pred_np_path)
                        np.save(right_pred_np_path, right_sample_np[j].transpose(1, 2, 0))

                if save_tar:
                    np.save(
                        os.path.join(inference_output_path, fn + "_left_tar.npy"),
                        left_batch_np[j].transpose(1, 2, 0),
                    )
                    np.save(
                        os.path.join(inference_output_path, fn + "_right_tar.npy"),
                        right_batch_np[j].transpose(1, 2, 0),
                    )

                num_samples += 1

            progress.update(task_total_id, advance=1)
            if not rgb_only:
                metric_value = metrics_sampling.compute()
                psnr_rgb = metric_value["psnr_rgb"].item()
                ssim_rgb = metric_value["ssim_rgb"].item()
                task_total.fields["psnr_rgb"] = f"{psnr_rgb:.3f}"
                task_total.fields["ssim_rgb"] = f"{ssim_rgb:.4f}"

    if not rgb_only:
        batch = {}
        for key, value in metrics_sampling.compute().items():
            print("%s: %.6f" % (key, value.item()))
            batch[key] = value.item()

        with open(metric_path, "w") as f:
            yaml.dump(batch, f)

        with open(config_path, "w") as f:
            yaml.dump(OmegaConf.to_container(cfg), f)

    print("sampling complete")


if __name__ == "__main__":
    main()
