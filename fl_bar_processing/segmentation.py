import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, label
from skimage import measure
from sklearn.cluster import KMeans

from fl_bar_processing.config import SEGMENTATION

from .core import FlBarImage, get_parent_dir_path


class FlBarSegmentation:
    count = 0

    def __init__(
        self,
        masked_color_image: "FlBarImage",
        timestamp: str = None,
    ) -> None:
        if masked_color_image.image_type != "masked_3-color":
            raise ValueError("masked_color_image must be a masked_3-color image")
        self.masked_color_image = masked_color_image
        self.label_map = None
        self.segmented_image_dataframe = pd.DataFrame()
        self.timestamp = timestamp

    def blue_kmeans_segmentation(self, c3_background: tuple = None) -> None:
        mask = self.masked_color_image.array.mean(axis=2) > 0
        plt.figure(figsize=(14, 4))
        blue_image = self.masked_color_image.array[:, :, 2]
        blue_image_log = np.log1p(blue_image)
        blue_image_log[np.isinf(blue_image_log)] = 0
        background_mask = np.zeros_like(blue_image)
        background_mask[
            (blue_image > c3_background[0]) & (blue_image < c3_background[1])
        ] = 1        

        reshaped_blue_image = blue_image_log.reshape(-1, 1)
        kmeans = KMeans(n_clusters=SEGMENTATION["n_clusters"], random_state=42)
        kmeans_labels = kmeans.fit_predict(reshaped_blue_image)
        segmented_image = kmeans_labels.reshape(blue_image_log.shape[:2])

        label_map = np.zeros_like(segmented_image, dtype=np.int32)
        new_label = 1

        layer_props = []
        angle_degs = []

        for cluster_id in range(np.max(segmented_image) + 1):
            cluster_mask = (segmented_image == cluster_id).astype(np.uint8)
            labeled_clusters, num_labels = label(cluster_mask)
            for label_ in range(1, num_labels + 1):
                component = (labeled_clusters == label_).astype(np.uint8)
                props = measure.regionprops(component.astype(np.int32))

                if len(props) == 0:
                    continue
                miny, minx, maxy, maxx = props[0].bbox
                height = maxy - miny
                width = maxx - minx

                # 長軸がx軸となす角
                angle_deg = abs(90 - abs(np.degrees(props[0].orientation)))
                angle_degs.append(angle_deg)

                area = props[0].area
                perimeter = props[0].perimeter
                y_min = props[0].bbox[0]

                circularity = (
                    4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                )
                filled = binary_fill_holes(component)
                has_holes = np.sum(filled) - np.sum(component) > 10
                if (
                    area >= 100
                    and circularity >= 0.3
                    and circularity < 0.9  # まんまるの蛍光ゴミを除く
                    and not has_holes
                ):
                    plt.figure()
                    if (
                        circularity < 0.9  # ゴミではなく
                        and height / width > SEGMENTATION["max_width_height_ratio"]
                    ):
                        plt.title(f"{height/width=}")
                        plt.imshow(labeled_clusters, cmap="jet")
                        return
                    elif angle_deg > SEGMENTATION["angle_threshold"]:
                        plt.title(f"{angle_deg=}")
                        plt.imshow(labeled_clusters, cmap="jet")
                        return
                    label_map[component > 0] = new_label
                    new_label += 1

        if new_label < 3:
            return
        
        # 穴の空いていない領域を水平に埋める
        for blue_label in range(1, new_label):
            y, x = np.where(label_map == blue_label)
            if len(y) == 0:
                return
            blue_y_min = np.min(y)
            blue_y_max = np.max(y)
            label_map[blue_y_min:blue_y_max, :] = blue_label
        label_map *= mask

        # Fill remaining largest areas until 5 layers are obtained
        remaining_mask = mask & (label_map == 0)
        labeled_remaining_mask, num_remaining_labels = label(remaining_mask)
        label_area_dict = {}
        for label_ in range(1, num_remaining_labels + 1):
            label_area_dict[label_] = np.sum(labeled_remaining_mask == label_)
        sorted_labels = sorted(
            label_area_dict.items(), key=lambda x: x[1], reverse=True   
        )
        # get largest areas to make total 5 layers
        for label_, area in sorted_labels[:5-(new_label - 1)]:
            if label_ == 0:
                continue
            label_map[labeled_remaining_mask == label_] = new_label
            y_min
            new_label += 1
        
        # Re-label layers based on their y-coordinate (top to bottom)
        label_y_dict = {}
        for final_label in np.unique(label_map):
            if final_label == 0:
                continue
            y, x = np.where(label_map == final_label)
            label_y_dict[final_label] = np.median(y)
        sorted_labels_by_y = sorted(
            label_y_dict.items(), key=lambda x: x[1]
        )
        new_label_map = np.zeros_like(label_map)
        for new_label, (old_label, _) in enumerate(sorted_labels_by_y, start=1):
            new_label_map[label_map == old_label] = new_label
        label_map = new_label_map

        if np.max(label_map) != 5:
            return
        
        # apply background mask to only odd layers
        is_odd = label_map % 2 == 1
        label_map_odd = label_map * is_odd * background_mask
        label_map_even = label_map * ~is_odd
        label_map = label_map_odd + label_map_even

        for layer_label in np.unique(label_map):
            if np.sum(label_map == layer_label) / np.sum(mask) < 0.03:
                return

        parent_dir = get_parent_dir_path(self.masked_color_image.image_path)
        self.label_map = FlBarImage(
            image_path=f"{parent_dir}/label_map.tif", array=label_map, image_type="gray"
        )
        self.label_map.save()

        FlBarSegmentation.count += 1
        plt.figure()
        # Display original segmentation
        plt.subplot(1, 3, 1)
        plt.imshow(blue_image_log, cmap="jet")
        plt.title("Blue image")

        # Display filtered segmentation
        plt.subplot(1, 3, 2)
        plt.imshow(label_map, cmap="jet")
        plt.title("Label map")

        # Display final output
        plt.subplot(1, 3, 3)
        plt.imshow(self.masked_color_image.array // 64, cmap="jet")
        plt.title("Original")

        plt.tight_layout()
        plt.savefig(f"{parent_dir}/result.png", bbox_inches="tight")

        plt.show()

        layer_props = {}
        mask_y_min = np.min(np.where(mask)[0])
        mask_y_max = np.max(np.where(mask)[0])

        for final_label in np.unique(label_map):
            layer_prop = {}
            if final_label == 0:
                continue
            y, x = np.where(label_map == final_label)
            y_center = np.median(y)
            x_center = np.median(x)
            for i in range(3):
                layer_prop[f"c{i + 1}_intensity_median"] = np.median(
                    self.masked_color_image.array[y, x, i]
                )
                layer_prop[f"c{i + 1}_intensity_mean"] = np.mean(
                    self.masked_color_image.array[y, x, i]
                )
                layer_prop[f"c{i + 1}_intensity_max"] = np.max(
                    self.masked_color_image.array[y, x, i]
                )
                layer_prop[f"c{i + 1}_intensity_min"] = np.min(
                    self.masked_color_image.array[y, x, i]
                )
                layer_prop[f"c{i + 1}_intensity_center_5pixel"] = np.mean(
                    self.masked_color_image.array[
                        int(y_center), int(x_center - 2) : int(x_center + 3), i
                    ]
                )
            layer_prop["y_center"] = y_center
            layer_prop["x_center"] = x_center
            layer_prop["y_center_ratio"] = (y_center - mask_y_min) / (
                mask_y_max - mask_y_min
            )
            # (np.mean(y) - min_y) / (max_y - min_y)
            layer_prop["bar_id"] = parent_dir.split("/")[-1]
            layer_prop["original_label"] = final_label
            layer_prop["layer area"] = np.sum(label_map == final_label)
            layer_prop["mask area"] = np.sum(mask)
            layer_prop["area_ratio"] = np.sum(label_map == final_label) / np.sum(mask)
            layer_props[final_label] = layer_prop

        if len(layer_props) < 5:
            return

        self.segmented_image_dataframe = pd.DataFrame(layer_props).T.sort_values(
            by="y_center_ratio"
        )

        self.segmented_image_dataframe["layer_id"] = [
            f"Layer {layer + 1}" for layer in range(5)
        ]

        layer_2_blue = self.segmented_image_dataframe[
            self.segmented_image_dataframe["layer_id"] == "Layer 2"
        ]["c3_intensity_center_5pixel"].mean()  # len = 1
        layer_4_blue = self.segmented_image_dataframe[
            self.segmented_image_dataframe["layer_id"] == "Layer 4"
        ]["c3_intensity_center_5pixel"].mean()

        if layer_4_blue > layer_2_blue:
            self.segmented_image_dataframe = self.segmented_image_dataframe.loc[::-1]
            self.segmented_image_dataframe["y_center_ratio"] = (
                1 - self.segmented_image_dataframe["y_center_ratio"]
            )
        self.segmented_image_dataframe["layer_id"] = [
            f"Layer {layer + 1}" for layer in range(5)
        ]
        self.segmented_image_dataframe["parent_folder"] = self.timestamp