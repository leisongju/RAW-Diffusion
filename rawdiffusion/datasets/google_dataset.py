import os
import numpy as np
import torch
import cv2
import glob




class GoogleDataset:
    def __init__(self, root_dir="/mnt/datalsj/dual_pixel/data/google", is_train=False, transforms=None):
        self.is_train = is_train
        if is_train:
            self.root_dir = os.path.join(root_dir, "train")
        else:
            self.root_dir = os.path.join(root_dir, "test")
        
        # 搜索 scaled_images 目录下的所有 jpg 文件
        self.image_paths = glob.glob(os.path.join(self.root_dir, "scaled_images", "**", "*.jpg"), recursive=True)
        self.image_paths = sorted(self.image_paths)
        self.transforms = transforms
    def __len__(self):
        return len(self.image_paths)

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2, 0, 1)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # 加载 RGB 图像
        rgb_image = cv2.imread(image_path) / 255.0

        # 获取相对路径和文件名
        rel_path = os.path.relpath(image_path, os.path.join(self.root_dir, "scaled_images"))
        dir_name = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # result_scaled_image_bottom
        
        raw_left_name = base_name.replace("scaled_image", "pd_left") + ".png"
        raw_right_name = base_name.replace("scaled_image", "pd_right") + ".png"
    
        # 构建完整路径
        raw_left_path = os.path.join(self.root_dir, "raw_left_pd", dir_name, raw_left_name)
        raw_right_path = os.path.join(self.root_dir, "raw_right_pd", dir_name, raw_right_name)
        
        # 加载对应的数据
        left_raw = cv2.imread(raw_left_path) / 255.0
        right_raw = cv2.imread(raw_right_path) / 255.0

        if not self.is_train:
            rgb_image = cv2.resize(rgb_image, (736, 992))

        if left_raw.shape[:2] != rgb_image.shape[:2]:
            left_raw = cv2.resize(left_raw, (rgb_image.shape[1], rgb_image.shape[0]))
        if right_raw.shape[:2] != rgb_image.shape[:2]:
            right_raw = cv2.resize(right_raw, (rgb_image.shape[1], rgb_image.shape[0])) 

        if self.transforms is not None:
            rgb_image, left_raw, right_raw = self.transforms(rgb_image, left_raw, right_raw)

        rgb_image = self.np2tensor(rgb_image).float()
        left_raw = self.np2tensor(left_raw).float()
        right_raw = self.np2tensor(right_raw).float()
        
        rgb_image = rgb_image * 2 - 1
        left_raw = left_raw * 2 - 1
        right_raw = right_raw * 2 - 1
        
        return {
            'guidance_data': rgb_image,
            'left_raw_data': left_raw,
            'right_raw_data': right_raw,
            'path': os.path.relpath(image_path, self.root_dir)
        }
    


if __name__ == "__main__":
    dataset = GoogleDataset()
    print(f"数据集大小: {len(dataset)}")
    
    # 测试第一个样本
    sample = dataset[0]
    print(f"RGB图像形状: {sample['rgb'].shape}")
    print(f"左眼原始PD形状: {sample['left_raw'].shape}")
    print(f"左眼PD结果形状: {sample['left_pd'].shape}")
    
    # 打印前几个样本的路径信息
    for i in range(min(3, len(dataset))):
        print(f"\n样本 {i}:")
        print(f"RGB路径: {dataset.image_paths[i]}")