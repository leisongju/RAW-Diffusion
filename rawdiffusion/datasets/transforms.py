import random

import numpy as np


class ImageTransforms:
    def __init__(self, patch_size, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train

    def random_flip(self, rgb_data, raw_left_data, raw_right_data):
        idx = np.random.randint(2)
        rgb_data = np.flip(rgb_data, axis=idx).copy()
        raw_left_data = np.flip(raw_left_data, axis=idx).copy()
        raw_right_data = np.flip(raw_right_data, axis=idx).copy()

        return rgb_data, raw_left_data, raw_right_data

    def random_rotate(self, rgb_data, raw_left_data, raw_right_data):
        idx = np.random.randint(4)
        rgb_data = np.rot90(rgb_data, k=idx)
        raw_left_data = np.rot90(raw_left_data, k=idx)
        raw_right_data = np.rot90(raw_right_data, k=idx)

        return rgb_data, raw_left_data, raw_right_data

    def random_crop(
        self,
        rgb_data,
        raw_left_data,
        raw_right_data,
    ):
        image_height, image_width, _ = raw_left_data.shape
        rnd_h = random.randint(0, max(0, image_height - self.patch_size))
        rnd_w = random.randint(0, max(0, image_width - self.patch_size))

        patch_input_raw = raw_left_data[
            rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :
        ]
        patch_rgb_data = rgb_data[
            rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :
        ]
        patch_raw_right_data = raw_right_data[
            rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :
        ]

        return patch_rgb_data, patch_input_raw, patch_raw_right_data

    def center_crop(self, rgb_data, raw_left_data, raw_right_data):
        image_height, image_width, _ = rgb_data.shape
        height_new = self.patch_size
        width_new = self.patch_size

        offset_y = (image_height - height_new) // 2
        offset_x = (image_width - width_new) // 2

        patch_input_raw = raw_left_data[
            offset_y : offset_y + height_new, offset_x : offset_x + width_new, :
        ]
        patch_rgb_data = rgb_data[
            offset_y : offset_y + height_new, offset_x : offset_x + width_new, :
        ]
        patch_raw_right_data = raw_right_data[
            offset_y : offset_y + height_new, offset_x : offset_x + width_new, :
        ]

        return patch_rgb_data, patch_input_raw, patch_raw_right_data

    def __call__(self, raw_left_data, rgb_data, raw_right_data):
        assert raw_left_data.shape[:2] == rgb_data.shape[:2]

        if self.is_train:
            rgb_data, raw_left_data, raw_right_data = self.random_crop(
                rgb_data,
                raw_left_data,
                raw_right_data,
            )
            rgb_data, raw_left_data, raw_right_data = self.random_rotate(
                rgb_data,
                raw_left_data,
                raw_right_data,
            )
            rgb_data, raw_left_data, raw_right_data = self.random_flip(
                rgb_data,
                raw_left_data,
                raw_right_data,
            )
        else:
            rgb_data, raw_left_data, raw_right_data = self.center_crop(
                rgb_data,
                raw_left_data,
                raw_right_data,
            )

        return raw_left_data, rgb_data, raw_right_data
