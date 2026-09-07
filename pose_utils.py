import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


POSE_COLUMNS = ["tx", "ty", "tz", "rx", "ry", "rz"]
DATA_COLUMNS = ["name", *POSE_COLUMNS]


def load_pose_table(path):
    table = pd.read_csv(path)

    missing = [column for column in DATA_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"Invalid pose dataset: missing columns {missing}. "
            "Regenerate camera-pose labels using the updated collection script."
        )

    poses = table[POSE_COLUMNS].to_numpy(dtype=np.float64)

    if not np.isfinite(poses).all():
        raise ValueError("Pose labels contain non-finite values.")

    return table[DATA_COLUMNS].to_numpy()


def matrix_to_pose(matrix):
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 4)
    translation = matrix[:, 3]
    rotation = Rotation.from_matrix(matrix[:, :3]).as_rotvec()
    return np.concatenate((translation, rotation))


def relative_pose(current_pose, desired_pose):
    current_pose = np.asarray(current_pose, dtype=np.float64)
    desired_pose = np.asarray(desired_pose, dtype=np.float64)

    current_rotation = Rotation.from_rotvec(current_pose[3:6]).as_matrix()
    desired_rotation = Rotation.from_rotvec(desired_pose[3:6]).as_matrix()

    relative_rotation = current_rotation.T @ desired_rotation
    relative_translation = current_rotation.T @ (
        desired_pose[:3] - current_pose[:3]
    )

    relative_rotation_vector = Rotation.from_matrix(
        relative_rotation
    ).as_rotvec()

    return np.concatenate(
        (relative_translation, relative_rotation_vector)
    ).astype(np.float32)