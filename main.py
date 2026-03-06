import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff

from fl_bar_processing import (
    BINARIZATION,
    COLOR,
    MASKING,
    PATH,
    SEGMENTATION,
    FlBarAlignment,
    FlBarSegmentation,
    Masking,
    SingleLayerProcessing,
    correct_color_misalignment,
    get_color_image,
    get_gray_image,
    load_and_predict,
)


@dataclass
class FieldConfig:
    well_name: str
    field_id: int
    flbar_lot: str
    date: str
    layer_type: Literal["5-layer", "single-layer"]
    layer_id: str | None = None
    grounded_truth: int | None = None


class FlBarProcessor:
    def __init__(
        self,
        color_image_file_path: str | List[str],
        gray_image_file_path: str,
        masked_image_file_path: str,
        field_config: 'FieldConfig',
        timestamp: str,
        blurred_field_list: List[str] | None = None,
        save_dir: bool = True,
    ) -> None:
        self.color_image_file_path = color_image_file_path
        self.gray_image_file_path = gray_image_file_path
        self.masked_image_file_path = masked_image_file_path
        self.field_config = field_config
        self.layer_type = field_config.layer_type
        self.layer_id = field_config.layer_id
        self.field_name = f"{field_config.well_name}_fld-{field_config.field_id}"
        self.timestamp = timestamp
        self.file_name = f"{timestamp}/{self.field_name}"
        self.dir = f"{PATH['save_dir_path']}/{self.field_name}"
        self.dataframe = None
        self.blurred_field_list = blurred_field_list
        self.save_dir = save_dir

    def process_field(
        self,
    ) -> None:
        color_image = get_color_image(self.color_image_file_path, self.dir)

        if self.blurred_field_list is None:
            focus_result, confidence = load_and_predict(
                image_path=self.gray_image_file_path,
            )
            if focus_result == "blurred":
                print(
                    f"Image {self.color_image_file_path[0].split('/')[-1]} is blurred with confidence {confidence}."
                )
                return
            else:
                print(
                    f"Image {self.color_image_file_path[0].split('/')[-1]} is focused with confidence {confidence}."
                )
        else:
            if self.field_name in self.blurred_field_list:
                return

        bright_field_image = get_gray_image(
            self.gray_image_file_path, self.dir)
        mask_image = tiff.imread(self.masked_image_file_path)

        color_image.array = correct_color_misalignment(
            target_image=color_image.array)

        dfs = []
        if self.field_config.layer_type == "5-layer":
            masking = Masking(color_image, bright_field_image, mask_image, 
                              background_n_sigma=MASKING["background_n_sigma"])
            masking.save_masked_image(f"{self.file_name}")
            for masked_color_image, c3_bc in zip(masking.masked_fl_bar_images, masking.c3_backgrounds):
                fl_bar_alignment = FlBarAlignment(masked_color_image)
                fl_bar_alignment.align_fl_bar_by_shape()
                fl_bar_alignment.aligned_image.save()

                fl_bar_segmentation = FlBarSegmentation(
                    fl_bar_alignment.aligned_image,
                    timestamp=self.timestamp,
                )
                fl_bar_segmentation.blue_kmeans_segmentation(
                    c3_bc)

                dfs.append(
                    fl_bar_segmentation.segmented_image_dataframe
                    
                )
            self.save_config()
            self.dataframe = self.get_concatenated_dataframe(
                dfs)
            if self.save_dir:
                self.save_field_dir()

        else:
            single_layer_processing = SingleLayerProcessing(
                mask_image=mask_image,
                color_image=color_image.array,
                ground_truth=self.field_config.grounded_truth,
                field_name=self.field_name,
                layer_id=self.field_config.layer_id,
            )
            self.save_config()
            self.dataframe = single_layer_processing.get_layer_props()
            self.save_field_dir()

    def save_field_dir(self) -> None:
        source_dir = f'/content/project-chocobar/notebooks/output/{self.timestamp}/{self.field_name}'
        dest_dir = f'/content/drive/MyDrive/output/{self.field_config.date}/{self.timestamp}/{self.field_name}'
        shutil.copytree(source_dir, dest_dir)

    def get_concatenated_dataframe(
        self,
        dfs: List[pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        if len(dfs) > 0:
            concatenated_dataframe = pd.concat(dfs, ignore_index=True)
            concatenated_dataframe.to_csv(
                f"{PATH['save_dir_path']}/{self.file_name}/result.csv",
                index=False,
            )
            concatenated_dataframe["field_name"] = [self.field_name] * len(
                concatenated_dataframe
            )
            return concatenated_dataframe
        else:
            return None

    def show_color_histogram(
        self, intensity: Literal["mean", "median", "max", "min"] = "median"
    ) -> None:
        for layer in range(1, 6):
            df_layer = self.dataframe[self.dataframe["layer_id"]
                                      == f"Layer {layer}"]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            channel_names = [
                "Red - Cy5",
                "Blue - FITC",
                "UV - DAPI",
            ]  # Names from file paths

            for i in range(1, 4):
                ax = axes[i - 1]
                # plot
                ax.set_xscale("log")
                ax.hist(
                    df_layer[f"c{i}_intensity_{intensity}"],
                    bins=np.logspace(0, 10, 50),
                    alpha=0.7,
                )
                ax.set_xlabel("Intensity")
                ax.set_ylabel("Frequency")
                ax.set_xlim(1, 1e6)
                ax.set_title(f"Channel {i}: {channel_names[i - 1]}")

            plt.tight_layout()
            plt.suptitle(f"Layer {layer} Intensity Histograms", fontsize=16)
            plt.subplots_adjust(top=0.85)  # Add some space for the suptitle
            plt.savefig(
                f"{PATH['save_dir_path']}/{self.file_name}/layer_{layer}_color_histogram.png"
            )

    def show_position_histogram(self) -> None:
        if self.layer_type == "single-layer":
            return
        else:
            plt.figure()
            for layer in range(1, 6):
                df_layer = self.dataframe[
                    self.dataframe["layer_id"] == f"Layer {layer}"
                ]
                plt.hist(
                    df_layer["y_center_ratio"],
                    bins=30,
                    alpha=0.7,
                    label=f"Layer {layer}",
                )
            plt.xlabel("Y center ratio")
            plt.ylabel("Frequency")
            plt.title("Y center ratio Distribution")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(
                f"{PATH['save_dir_path']}/{self.file_name}/layer_position_histogram.png"
            )

    def save_config(self) -> None:
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        config = {
            "BINARIZATION": make_json_serializable(BINARIZATION),
            "MASKING": make_json_serializable(MASKING),
            "PATH": make_json_serializable(PATH),
            "COLOR": make_json_serializable(COLOR),
            "SEGMENTATION": make_json_serializable(SEGMENTATION),
        }
        config["Git Commit ID"] = get_git_commit_id()

        with open(
            f"{PATH['save_dir_path']}/{self.file_name}/param_config.json", "w"
        ) as f:
            json.dump(config, f, indent=4)


def process_well(
        cloud_path: str,
        well_name: str,
        fields: int,
        layer_type: Literal["5-layer", "single-layer"],
        flbar_lot: str,
        date: str,
) -> tuple[str, List[FlBarProcessor]]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    processors = []
    for field in fields:
        filed_config = FieldConfig(
            well_name=well_name,
            field_id=field,
            layer_type=layer_type,
            flbar_lot=flbar_lot,
            date=date
        )
        # try:
        channel_names = [
            "Red - Cy5",
            "Blue - FITC",
            "UV - DAPI",
            "TL-Brightfield - dsRed"
        ]

        image_file_names = [rf"{cloud_path}/{filed_config.well_name}(fld {field} wv {channel_name}).tif" \
                            for channel_name in channel_names]

        processor = FlBarProcessor(
            color_image_file_path=image_file_names[:3],
            gray_image_file_path=image_file_names[3],
            masked_image_file_path=image_file_names[3].replace(".tif", "_mask.tif"),
            field_config=filed_config,
            timestamp=timestamp,
        )
        print(f"{processor.file_name} is processing...")
        processor.process_field()

        field_config_json = filed_config.__dict__

        if processor.dataframe is not None:
            with open(f"notebooks/output/{processor.file_name}/field_config.json", "w") as json_file:
                json.dump(field_config_json, json_file, indent=4)
            # processor.dataframe.to_csv(
            #     f"notebooks/output/{processor.file_name}/field_result.csv", index=False)
            try:
                processor.show_color_histogram()
                processor.show_position_histogram()

                processors.append(processor)
            except KeyError as e:
                print(f"KeyError encountered while processing {processor.file_name}: {e}")
                with open(f"notebooks/output/{processor.file_name}/error_log.txt", "w") as log_file:
                    log_file.write(f"KeyError: {e}\n")

    return timestamp, processors

def get_git_commit_id():
    try:
        commit_id = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
        return commit_id
    except subprocess.Calledprocessorror:
        return "unknown_commit_id"


def show_histogram(
    data: pd.DataFrame,
    x: Literal["color_intensities", "y_center_ratio", "area"],
    hue: Literal["color_intensities", "layer_id",
                 "field_id", "date"] | None = None,
    intensity: Literal["mean", "median", "min",
                       "max", "center_5pixel"] = "median",
    overlay: bool = True,
) -> None:
    plot_prop = {
        "data": data,
        "x": x,
        "hue": hue,
    }
    if hue is None:
        plot_prop = {
            "data": data,
            "x": x,
        }

    if x == "color_intensities":
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        for i in range(3):
            plot_prop["x"] = f"c{i + 1}_intensity_{intensity}"
            sns.histplot(
                **plot_prop,
                bins=50,
                stat="density",
                common_norm=False,
                kde=True,
                ax=axes[i],
                log_scale=(True, False),
            )
        axes[i].set_title(f"c{i + 1}_intensity Distribution by {hue}")

        # Set the same x-axis limits for all three subplots
        for c in range(1, 4):
            # Get min and max values for each channel across all data
            min_val = data[f"c{c}_intensity_{intensity}"].min()
            max_val = data[f"c{c}_intensity_{intensity}"].max()
            # Add some padding
            axes[c - 1].set_xlim(min_val, max_val * 1.05)

    elif x == "y_center_ratio":
        plot_prop["hue"] = "layer_id"
        plt.figure()
        sns.histplot(
            **plot_prop,
            bins=50,
            stat="density",
            common_norm=False,
            kde=True,
            log_scale=(False, False),
        )
        plt.title(f"y_center_ratio Distribution by {hue}")

    elif x == "area_ratio":
        if overlay:
            plot_prop["hue"] = "layer_id"
            plt.figure()
            sns.histplot(
                **plot_prop,
                bins=50,
                stat="density",
                common_norm=False,
                kde=True,
                log_scale=(False, False),
            )
            plt.title(f"Area Distribution by {hue}")
        else:
            fig, axes = plt.subplots(5, 1, figsize=(12, 18))
            for i in range(1, 6):
                plot_prop["data"] = data[data["layer_id"] == f"Layer {i}"]
                sns.histplot(
                    **plot_prop,
                    bins=50,
                    stat="density",
                    common_norm=False,
                    kde=True,
                    ax=axes[i - 1],
                    log_scale=(False, False),
                )
                axes[i - 1].set_title(f"Area Distribution by {hue} Layer {i}")

    plt.tight_layout()


def show_layer_color_histogram(
    data: pd.DataFrame,
    hue: Literal["layer_id", "field_id"] | None = None,
    intensity: Literal["mean", "median", "min",
                       "max", "center_5pixel"] = "median",
) -> None:
    if hue is None:
        plot_prop = {}
    plot_prop = {"hue": hue}
    plt.figure(figsize=(18, 12))

    j = 1
    for i in range(1, 6):
        plot_prop["data"] = data[data["layer_id"] == f"Layer {i}"]
        for c in range(1, 4):
            plot_prop["x"] = f"c{c}_intensity_{intensity}"
            plt.subplot(5, 3, j)

            sns.histplot(
                **plot_prop,
                bins=50,
                stat="density",
                common_norm=False,
                kde=False,
                log_scale=(True, False),
                palette="tab10",
            )
            plt.title(f"Layer {i} c{c}_intensity_{intensity} Distribution")
            j += 1
    plt.tight_layout()

    # Apply limits to all subplots
    for ax_num in range(1, 16):
        ax = plt.subplot(5, 3, ax_num)
        ax.set_xlim([50, 65535])
