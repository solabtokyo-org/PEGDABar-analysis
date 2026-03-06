from dataclasses import dataclass
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


@dataclass
class FlBarImage:
    image_path: str | List[str]
    array: np.ndarray
    image_type: Literal[
        "normalized_3-color", "3-color", "gray", "binary", "masked_3-color"
    ]  

    def save(self) -> None:
        if not isinstance(self.array, np.ndarray):
            return
        if isinstance(self.image_path, list):
            for i, path in enumerate(self.image_path):
                tiff.imwrite(path, self.array[:, :, i])
        else:
            tiff.imwrite(self.image_path, self.array)


def get_color_image(image_path: str | List[str], save_path: str) -> "FlBarImage":
    if isinstance(image_path, list) and len(image_path) == 3:
        image_c1 = tiff.imread(image_path[0])
        image_c2 = tiff.imread(image_path[1])
        image_c3 = tiff.imread(image_path[2])
    elif isinstance(image_path, str):
        return FlBarImage(
            image_path=save_path, array=tiff.imread(image_path), image_type="3-color"
        )
    else:
        raise ValueError(
            "image_paths must be either a string or a list of three strings."
        )

    img_16bit_3c = np.zeros((image_c1.shape[1], image_c1.shape[0], 3), dtype=np.uint16)
    img_16bit_3c[:, :, 0] = image_c1
    img_16bit_3c[:, :, 1] = image_c2
    img_16bit_3c[:, :, 2] = image_c3

    return FlBarImage(image_path=save_path, array=img_16bit_3c, image_type="3-color")


def get_gray_image(image_path: str, save_path: str) -> "FlBarImage":
    save_path = save_path + "/gray_image.tif"
    return FlBarImage(
        image_path=save_path, array=tiff.imread(image_path), image_type="gray"
    )


def color2gray(color_image: "FlBarImage", image_path="") -> "FlBarImage":
    if (color_image.image_type != "3-color") and (
        color_image.image_type != "masked_3-color"
    ):
        raise ValueError("Input image must be of type '3-color'.")
    return FlBarImage(
        image_path=image_path,
        array=np.sum(color_image.array, axis=2) // 3,
        image_type="gray",
    )


def get_c3_intensity_boundary(array: np.ndarray, n_sigma: int) -> tuple[int, int]:
    flattened_array = array.flatten()
    plt.figure(figsize=(6, 2))
    plt.hist(flattened_array, bins=1000, color="blue", alpha=0.7)
    mean_value = np.mean(flattened_array)
    std_dev = np.std(flattened_array)
    lower_bound = mean_value - n_sigma * std_dev
    upper_bound = mean_value + n_sigma * std_dev
    plt.axvline(mean_value, color="green", linestyle="dashed", linewidth=1)
    plt.axvline(lower_bound, color="red", linestyle="dashed", linewidth=1)
    plt.axvline(upper_bound, color="red", linestyle="dashed", linewidth =1)
    plt.title("C3 Intensity Distribution with N-Sigma Bounds")
    return (max(lower_bound, 0), min(upper_bound, 65535))

def get_parent_dir_path(image_path: str) -> str:
    return "/".join(image_path.split("/")[:-1])
