from pathlib import Path

import numpy as np
from PIL import Image


def calculate_mae_rmse(
    image_path1: str,
    image_path2: str,
) -> tuple[float, float]:
    path1 = Path(image_path1)
    path2 = Path(image_path2)

    if not path1.is_file():
        raise FileNotFoundError(f"error：{path1}")

    if not path2.is_file():
        raise FileNotFoundError(f"error：{path2}")

    with Image.open(path1) as image1:
        image1 = np.asarray(
            image1.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

    with Image.open(path2) as image2:
        image2 = np.asarray(
            image2.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

    if image1.shape != image2.shape:
        raise ValueError(
            f"error：{image1.shape} and {image2.shape}"
        )

    error = image1 - image2

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))

    return mae, rmse



mae, rmse = calculate_mae_rmse(
    "prediction.png",
    "ground_truth.png",
)

print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")