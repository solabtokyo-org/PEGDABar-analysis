import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from fl_bar_processing.config import MASKING, PATH
from fl_bar_processing.core import FlBarImage, get_c3_intensity_boundary


def generate_random_colors(num_colors):
    random_colors = [
        (random.random(), random.random(), random.random())
        for _ in range(num_colors + 1)
    ]
    random_colors[0] = (0, 0, 0)
    return ListedColormap(random_colors)


class Masking:
    def __init__(
        self,
        color_image: "FlBarImage",
        bright_field_image: "FlBarImage",
        labeled_image: np.ndarray = None,
        background_n_sigma: int = 15,
    ) -> None:
        if color_image.image_type != "3-color":
            raise ValueError("color_image must be a 3-color image")

        if bright_field_image.image_type != "gray":
            raise ValueError("bright_field_image must be a gray image")

        self.color_image = color_image
        self.bright_field_image = bright_field_image  # 16-bit gray image, 2048 x 2048
        self.labeled_image = labeled_image  # labeled image, 8-bit, 2048 x 2048
        self.background_n_sigma = background_n_sigma
        self.masked_fl_bar_images = []
        self.area_ids = []
        self.c3_backgrounds = []
        self.c3_backgrounds = []

    def save_masked_image(
        self,
        image_file_name: str,
        save_dir_path=PATH["save_dir_path"],
    ) -> None:
        self.labeled_image = cv2.dilate(
            self.labeled_image, np.ones((3, 3), np.uint8), iterations=3
        )
        background_mask = self.labeled_image == 0 # 2048 x 2048
        random_colors = generate_random_colors(
            len(np.unique(self.labeled_image)) - 1
        )  # -1 to exclude background
        plt.figure(figsize=(30, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(self.bright_field_image.array, cmap="gray")
        plt.title("Bright Field Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Labeled Image")
        plt.imshow(self.labeled_image, cmap=random_colors)
        plt.axis("off")

        for i, area_id in enumerate(np.unique(self.labeled_image)):
            # Skip the background area
            if area_id == 0:
                continue

            dir_name = f"{save_dir_path}/{image_file_name}/{i}"
            os.makedirs(dir_name, exist_ok=True)

            ellipse_mask = self.labeled_image == area_id

            masked_color_image = FlBarImage(
                image_path=f"{dir_name}/masked_color_image_{i}.tif",
                array=self.color_image.array.copy(),
                image_type="masked_3-color",
            )  # masked color image 2048x2048: finally to be cropped and masked

            raw_color_image = FlBarImage(
                image_path=f"{dir_name}/raw_color_image_{i}.tif",
                array=self.color_image.array.copy(),
                image_type="3-color",
            )  # raw color image 2048x2048: finally to be cropped

            xs, ys = np.where(self.labeled_image == area_id)
            min_x, max_x = np.min(xs), np.max(xs)
            min_y, max_y = np.min(ys), np.max(ys)

            min_x = max(min_x - MASKING["crop_margin"], 0)
            min_y = max(min_y - MASKING["crop_margin"], 0)
            max_x = min(max_x + MASKING["crop_margin"], self.labeled_image.shape[0] - 1)
            max_y = min(max_y + MASKING["crop_margin"], self.labeled_image.shape[1] - 1)

            if (
                (min_x == 0)
                or (min_y == 0)
                or (max_x == self.labeled_image.shape[0] - 1)
                or (max_y == self.labeled_image.shape[1] - 1)
            ):
                continue

            cropped_bright_field_image = FlBarImage(
                image_path=f"{dir_name}/cropped_bright_field_image_{i}.tif",
                array=self.bright_field_image.array[
                    min_x : max_x + 1, min_y : max_y + 1
                ],
                image_type="gray",
            )

            masked_color_image.array = masked_color_image.array[
                min_x : max_x + 1, min_y : max_y + 1, :
            ]

            raw_color_image.array = raw_color_image.array[
                min_x : max_x + 1, min_y : max_y + 1, :
            ]

            cropped_ellipse_mask = ellipse_mask[min_x : max_x + 1, min_y : max_y + 1]
            for c in range(3):
                channel = masked_color_image.array[:, :, c]
                # Zero out everything except the current object
                channel[cropped_ellipse_mask == 0] = 0
                masked_color_image.array[:, :, c] = channel
            
            cropped_background_mask = background_mask[
                min_x : max_x + 1, min_y : max_y + 1
            ]
            
            self.c3_backgrounds.append(get_c3_intensity_boundary(
                self.color_image.array[min_x : max_x + 1, min_y : max_y + 1, 2][
                    cropped_background_mask == 1
                ],
                n_sigma=self.background_n_sigma,
            ))

            masked_color_image.save()
            raw_color_image.save()
            cropped_bright_field_image.save()
            self.masked_fl_bar_images.append(masked_color_image)
            self.area_ids.append(area_id)
