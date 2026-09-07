import numpy as np
import matplotlib.pyplot as plt


def generate_cube_point(
    translation_range=0.2,
    rotation_range=1.0,
    num_samples=10000,
):
    translation = np.random.uniform(
        -translation_range,
        translation_range,
        size=(num_samples, 3),
    )

    rotation = np.random.uniform(
        -rotation_range,
        rotation_range,
        size=(num_samples, 3),
    )

    return np.concatenate((translation, rotation), axis=1)


def draw_scatter(points):
    fig = plt.figure(dpi=200, constrained_layout=True)
    plt.rcParams["font.family"] = "Times New Roman"

    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        for tick in axis.get_majorticklabels():
            tick.set_fontweight("bold")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="r",
        marker="o",
    )

    plt.show()


if __name__ == "__main__":
    points = generate_cube_point(
        translation_range=0.2,
        rotation_range=1.0,
        num_samples=10000,
    )
    draw_scatter(points)