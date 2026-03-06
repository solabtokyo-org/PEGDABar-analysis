from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


@dataclass
class LayerProp:
    c1_intensity_median: np.float64
    c1_intensity_mean: np.float64
    c1_intensity_max: np.float64
    c1_intensity_min: np.float64
    c1_intensity_center_5pixel: np.float64
    c2_intensity_median: np.float64
    c2_intensity_mean: np.float64
    c2_intensity_max: np.float64
    c2_intensity_min: np.float64
    c2_intensity_center_5pixel: np.float64
    c3_intensity_median: np.float64
    c3_intensity_mean: np.float64
    c3_intensity_max: np.float64
    c3_intensity_min: np.float64
    c3_intensity_center_5pixel: np.float64
    bar_id: int
    area: int
    layer_id: str
    field_name: str
    GroundTruth: int


class SingleLayerProcessing:
    def __init__(self,
                 mask_image: np.ndarray,
                 color_image: np.ndarray,
                 ground_truth: int,
                 field_name: str,
                 layer_id: str) -> None:
        self.mask_image = mask_image
        self.color_image = color_image
        self.ground_truth = ground_truth
        self.field_name = field_name
        self.layer_id = layer_id

    def filter_bar_mask(self) -> np.ndarray:

        labeled_mask = label(self.img_mask)
        regions = regionprops(labeled_mask)

        filtered_mask = np.zeros_like(self.img_mask)

        for region in regions:
            if (region.axis_minor_length > 0) and (region.axis_major_length/region.axis_minor_length < 3) and (region.area > 2500):
                filtered_mask[labeled_mask == region.label] = region.label
        self.img_mask = filtered_mask

    def get_layer_props(self) -> pd.DataFrame:
        self.filter_bar_mask()
        img_c1 = self.color_image[:, :, 0]
        img_c2 = self.color_image[:, :, 1]
        img_c3 = self.color_image[:, :, 2]
        masked_img_c3 = img_c3 * (self.img_mask > 0)

        masked_values_log = np.log1p(masked_img_c3[masked_img_c3 > 0])
        otsu_threshold = threshold_otsu(masked_values_log)

        threshold_mask = np.log1p(img_c3) > otsu_threshold
        layer_mask = self.img_mask * threshold_mask

        layer_props = []
        for mask_id in np.unique(layer_mask):
            if mask_id == 0:
                continue  # Skip background
            ys, xs = np.where(layer_mask == mask_id)
            min_x, max_x = np.min(xs), np.max(xs)
            min_y, max_y = np.min(ys), np.max(ys)
            if min_x == 0 or max_x == layer_mask.shape[1] - 1 or min_y == 0 or max_y == layer_mask.shape[0] - 1:
                continue  # Skip touching edges
            area = np.sum(layer_mask == mask_id)
            if area < 100:
                continue  # Skip small areas

            cropped_layer = layer_mask[min_y:max_y+1, min_x:max_x+1]
            cropped_img_c1 = img_c1[min_y:max_y+1, min_x:max_x+1]
            cropped_img_c2 = img_c2[min_y:max_y+1, min_x:max_x+1]
            cropped_img_c3 = img_c3[min_y:max_y+1, min_x:max_x+1]

            min_area = float('inf')
            best_angle = 0
            for angle in range(0, 180):
                rotated_layer = rotate(
                    cropped_layer, angle, reshape=True, order=0)
                ys, xs = np.where(rotated_layer > 0)
                _area = (np.max(ys) - np.min(ys) + 1) * \
                    (np.max(xs) - np.min(xs) + 1)
                if _area < min_area:
                    min_area = _area
                    best_angle = angle

            best_cropped_layer = rotate(
                cropped_layer, best_angle, reshape=True, order=0)
            best_ys, best_xs = np.where(best_cropped_layer > 0)
            height = np.max(best_ys) - np.min(best_ys)
            width = np.max(best_xs) - np.min(best_xs)
            if height > width:
                best_cropped_layer = best_cropped_layer.T
                best_angle += 90
                best_ys, best_xs = best_xs, best_ys

            center_x = np.mean(best_xs).astype(int)
            center_y = np.mean(best_ys).astype(int)

            best_cropped_img_c1 = rotate(
                cropped_img_c1, best_angle, reshape=True, order=1)
            best_cropped_img_c2 = rotate(
                cropped_img_c2, best_angle, reshape=True, order=1)
            best_cropped_img_c3 = rotate(
                cropped_img_c3, best_angle, reshape=True, order=1)

            layer_prop = LayerProp(
                c1_intensity_max=np.max(
                    best_cropped_img_c1[best_cropped_img_c1 > 0]),
                c1_intensity_min=np.min(
                    best_cropped_img_c1[best_cropped_img_c1 > 0]),
                c1_intensity_mean=np.mean(
                    best_cropped_img_c1[best_cropped_img_c1 > 0]),
                c1_intensity_median=np.median(
                    best_cropped_img_c1[best_cropped_img_c1 > 0]),
                c1_intensity_center_5pixel=np.mean(
                    best_cropped_img_c1[center_y, center_x-2:center_x+3]),
                c2_intensity_max=np.max(
                    best_cropped_img_c2[best_cropped_img_c2 > 0]),
                c2_intensity_min=np.min(
                    best_cropped_img_c2[best_cropped_img_c2 > 0]),
                c2_intensity_mean=np.mean(
                    best_cropped_img_c2[best_cropped_img_c2 > 0]),
                c2_intensity_median=np.median(
                    best_cropped_img_c2[best_cropped_img_c2 > 0]),
                c2_intensity_center_5pixel=np.mean(
                    best_cropped_img_c2[center_y, center_x-2:center_x+3]),
                c3_intensity_max=np.max(
                    best_cropped_img_c3[best_cropped_img_c3 > 0]),
                c3_intensity_min=np.min(
                    best_cropped_img_c3[best_cropped_img_c3 > 0]),
                c3_intensity_mean=np.mean(
                    best_cropped_img_c3[best_cropped_img_c3 > 0]),
                c3_intensity_median=np.median(
                    best_cropped_img_c3[best_cropped_img_c3 > 0]),
                c3_intensity_center_5pixel=np.mean(
                    best_cropped_img_c3[center_y, center_x-2:center_x+3]),
                bar_id=mask_id,
                area=area,
                layer_id=self.layer_id,
                field_name=self.field_name,
                GroundTruth=self.ground_truth
            )

            layer_props.append(layer_prop)

        return pd.DataFrame(
            [prop.__dict__ for prop in layer_props])
