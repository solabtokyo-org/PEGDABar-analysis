import cv2
import numpy as np
from scipy.ndimage import rotate

from .core import (
    FlBarImage,
    color2gray,
    get_parent_dir_path,
)


def vectorized_normalize_channels(data, epsilon=1e-10):
    min_vals = data.min(axis=(0, 1), keepdims=True)
    max_vals = data.max(axis=(0, 1), keepdims=True)
    delta = np.maximum(max_vals - min_vals, epsilon)
    return (data - min_vals) / delta


def get_ellipse_angle(mask):
    if mask.max() > 1:
        binary_mask = (mask > 0).astype(np.uint8)
    else:
        binary_mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        return None, None

    ellipse_params = cv2.fitEllipse(largest_contour)

    center, axes, angle = ellipse_params

    major_axis, minor_axis = axes[0], axes[1]

    if major_axis < minor_axis:
        vertical_angle = angle
    else:
        vertical_angle = (angle + 90) % 180

    return vertical_angle


class FlBarAlignment:
    def __init__(
        self,
        masked_color_image: "FlBarImage",
    ) -> None:
        if masked_color_image.image_type != "masked_3-color":
            raise ValueError("masked_color_image must be a masked_3-color image")

        self.masked_color_image = masked_color_image

        aligned_image_path = f"{get_parent_dir_path(self.masked_color_image.image_path)}/aligned_image.tif"

        self.aligned_image: "FlBarImage" = FlBarImage(
            aligned_image_path, None, "masked_3-color"
        )

    def align_fl_bar_by_shape(self) -> None:
        gray = color2gray(self.masked_color_image)
        binary = (gray.array > 0).astype(np.uint8)
        min_angle = -90
        max_angle = 90
        angle_step = 1.0
        angles = np.arange(min_angle, max_angle + angle_step, angle_step)
        ratios = []

        for angle in angles:
            rotated_binary = rotate(
                binary,
                angle,
                reshape=True,
                order=1,
                mode="constant",
                cval=0,
            )
            # Vertical projection (mean value)
            # Consider only the masked part
            horizontal_sum = np.sum(rotated_binary, axis=1)
            horizontal_counts = np.sum(horizontal_sum > 0)
            ratios.append(horizontal_counts)

        best_angle = angles[np.argmax(ratios)]

        mask = self.masked_color_image.array.mean(axis=2) > 0

        best_angle = get_ellipse_angle(mask)

        self.aligned_image.array = rotate(
            self.masked_color_image.array,
            best_angle,
            reshape=True,
            order=1,
            mode="constant",
            cval=0,
        )
