import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import point_generate
from pose_utils import DATA_COLUMNS, matrix_to_pose


def main():
    data_root = Path("./dataset/10000_camera_pose")
    image_root = data_root / "Image"
    label_path = data_root / "label.csv"

    if label_path.exists() or (
        image_root.exists() and any(image_root.iterdir())
    ):
        raise FileExistsError(
            f"Output dataset already exists: {data_root}. "
            "Use an empty output directory."
        )

    image_root.mkdir(parents=True, exist_ok=True)

    client = RemoteAPIClient()
    sim = client.require("sim")

    base_handle = sim.getObject("/UR5")
    target_handle = sim.getObject("/UR5/target")
    tip_handle = sim.getObject("/UR5/tip")
    sensor_handle = sim.getObject("/UR5/Vision_sensor")

    origin_position = sim.getObjectPosition(tip_handle)
    origin_orientation = sim.getObjectOrientation(tip_handle)
    original_target_matrix = sim.getObjectMatrix(target_handle)

    points = point_generate.generate_cube_point(
        translation_range=0.2,
        rotation_range=1.0,
        num_samples=10000,
    )

    try:
        with label_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(DATA_COLUMNS)

            for count, point in enumerate(points, start=1):
                new_position = [
                    origin_position[index] + point[index]
                    for index in range(3)
                ]
                new_orientation = [
                    origin_orientation[index] + point[index + 3]
                    for index in range(3)
                ]

                target_matrix = sim.buildMatrix(
                    new_position,
                    new_orientation,
                )
                sim.setObjectMatrix(target_handle, target_matrix)
                time.sleep(0.4)

                was_running = (
                    sim.getSimulationState()
                    == sim.simulation_advancing_running
                )

                if was_running:
                    sim.pauseSimulation()
                    while (
                        sim.getSimulationState()
                        != sim.simulation_paused
                    ):
                        time.sleep(0.005)

                try:
                    camera_matrix = sim.getObjectMatrix(
                        sensor_handle,
                        base_handle,
                    )
                    pose = matrix_to_pose(camera_matrix)

                    image_data, resolution = sim.getVisionSensorImg(
                        sensor_handle
                    )
                    image_array = np.frombuffer(
                        image_data,
                        dtype=np.uint8,
                    ).reshape(resolution[1], resolution[0], 3)

                    image = Image.fromarray(
                        np.flipud(image_array).copy()
                    )

                    image_name = f"{count}.png"
                    image.save(image_root / image_name)
                    writer.writerow([image_name, *pose.tolist()])
                    file.flush()
                finally:
                    if was_running:
                        sim.startSimulation()

                print(f"Image Name: {image_name}")
    finally:
        sim.setObjectMatrix(target_handle, original_target_matrix)


if __name__ == "__main__":
    main()