import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from pystackreg import StackReg

from fl_bar_processing.config import CALIBRATION


def calculate_transform_matrices(
    reference_image,
) -> tuple:
    visualize_misalignment(reference_image)

    ref_r, ref_g, ref_b = (
        reference_image[:, :, 0],
        reference_image[:, :, 1],
        reference_image[:, :, 2],
    )

    sr = StackReg(StackReg.AFFINE)
    r_matrix = sr.register(ref_g, ref_r)
    b_matrix = sr.register(ref_g, ref_b)

    return r_matrix, b_matrix


def save_transform_matrices(r_matrix, b_matrix, filename):
    matrices = {"r_matrix": r_matrix.tolist(), "b_matrix": b_matrix.tolist()}
    with open(filename, "w") as f:
        json.dump(matrices, f)


def load_transform_matrices(filename):
    if not os.path.exists(filename):
        return None, None
    with open(filename, "r") as f:
        matrices = json.load(f)
    r_matrix = np.array(matrices["r_matrix"])
    b_matrix = np.array(matrices["b_matrix"])

    return r_matrix, b_matrix


def correct_color_misalignment(
    target_image: np.ndarray,
    transform_matrices_path=CALIBRATION["transform_matrices_path"],
    reference_image_path=CALIBRATION["reference_image_path"],
) -> np.ndarray:
    target_r, target_g, target_b = (
        target_image[:, :, 0],
        target_image[:, :, 1],
        target_image[:, :, 2],
    )

    r_matrix, b_matrix = load_transform_matrices(transform_matrices_path)

    if r_matrix is None or b_matrix is None:
        images = []
        for _, path in reference_image_path.items():
            images.append(tifffile.imread(path))
        reference_image = np.stack(images, axis=-1)
        r_matrix, b_matrix = calculate_transform_matrices(reference_image)
        save_transform_matrices(r_matrix, b_matrix, transform_matrices_path)

    sr = StackReg(StackReg.AFFINE)

    r_corrected = sr.transform(target_r, tmat=r_matrix).astype(np.uint16)
    b_corrected = sr.transform(target_b, tmat=b_matrix).astype(np.uint16)

    return np.stack((r_corrected, target_g, b_corrected), axis=2)  # R=a, G=b, B=a の例


def visualize_misalignment(image: np.ndarray):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg_diff = np.abs(r - g).astype(np.uint8)
    rb_diff = np.abs(r - b).astype(np.uint8)
    gb_diff = np.abs(g - b).astype(np.uint8)

    plt.figure()
    plt.imshow(np.stack((rg_diff, rb_diff, gb_diff), axis=2))
