import os

working_dir = os.path.dirname(os.path.abspath(__file__))
print(f"working_dir: {working_dir}")
parent_dir_path = os.path.dirname(working_dir)

FOCUS_CHECK = {
    "model_path": f"{parent_dir_path}/focus_classifier_results/models/focus_classifier.pth",
}

CALIBRATION = {
    "reference_image_path": {
        "reference_c1_image_path": f"{parent_dir_path}/fl_bar_processing/calibration/Cy5_f1.tif",
        "reference_c2_image_path": f"{parent_dir_path}/fl_bar_processing/calibration/FITC_f1.tif",
        "reference_c3_image_path": f"{parent_dir_path}/fl_bar_processing/calibration/DAPI_f1.tif",
    },
    "transform_matrices_path": f"{parent_dir_path}/fl_bar_processing/calibration/transform_matrices.json",
}

BINARIZATION = {
    "median_blur_size": 51,  # ガウシアンブラーのカーネルサイズ
    "c1_threshold": 170,  # 二値化の閾値 (修正splitpool, c1 image)
    "c2_threshold": 130,  # 二値化の閾値 (修正splitpool, c2 image)
    "c3_threshold": 245,  # 二値化の閾値 (修正splitpool, c3 image)
    "threshold": 175,  # 二値化の閾値 (修正splitpool)
    # "threshold": 200,  # 二値化の閾値 (新しいsplitpool)
    # "threshold": 210,  # 二値化の閾値 (最初にもらった画像, 新しいsingle-layer A3)
    # "threshold": 2000,  # 二値化の閾値 (新しいsingle-layer A1, A2, A4)
    # "threshold": 800,  # 二値化の閾値 (新しいsingle-layer A5)
    # "threshold": 300,  # 二値化の閾値 (新しいsingle-layer A6)
    "similarity_threshold": 33,  # コサイン類似度の閾値
}


MASKING = {
    "crop_margin": 20,  # マスク画像の切り取り余白
    "removing_spot_area": 25,  # スポットの除去のための面積
    "bf_threshold": 580,  # BF画像の二値化の閾値
    "single_layer_min_area": 500,  # 面積の最小値
    "single_layer_max_area": 2000,  # 面積の最大値
    "single_bar_min_area": 1000,  # 面積の最小値
    "single_bar_max_area": 6000,  # 面積の最大値
    "peak_min_distance": 35,  # ピークの最小距離
    "peak_threshold_abs": 10,  # ピークの閾値
    "background_n_sigma": 15,  # バックグラウンドのNシグマ
    # "single_layer_min_area": 20,  # 面積の最小値
}

PATH = {
    "save_dir_path": rf"{parent_dir_path}/notebooks/output",  # マスク画像の保存先
}

COLOR = {"c1": "647", "c2": "488", "c3": "405"}

SEGMENTATION = {
    "layer_number": 5,
    # "layer_color_threshold": 0.1 * (2**16 - 1),
    "layer_color_threshold": 6600,
    "max_perimeter_area_ratio": 0.1,
    "min_circularity": 0.4,
    # "peak_height": 1.1,
    "peak_height": 0.3,
    "n_clusters": 6,  # KMeansのクラスタ数
    "max_width_height_ratio": 2,
    "angle_threshold": 30,  # 角度の閾値 (傾いているのを除く)
}
