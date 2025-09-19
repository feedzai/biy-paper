import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pyod.utils.data import generate_data
from scipy.stats import qmc
from shapely.geometry import Point, Polygon
from sklearn.datasets import make_blobs

from constants import DATASETS, NONE_LABEL
from utils import ensure_clean_dir, rotate_coordinates_around_center


def generate_gaussian_blobs(output_folder: Path) -> None:
    prefix = "gaussian_blobs"
    prefix_noise = f"{prefix}_noise"

    rng = np.random.default_rng(seed=0)

    min_n_clusters = 2
    max_n_clusters = 6

    for n_samples in [100, 500, 1_000, 2_000]:
        for n_clusters in range(min_n_clusters, max_n_clusters + 1):
            for cluster_std in [0.2, 0.4, 0.6]:
                coordinates, labels = make_blobs(
                    n_samples=n_samples, centers=n_clusters, cluster_std=cluster_std, random_state=0
                )

                dataset = pd.DataFrame(coordinates, columns=["x", "y"])
                dataset = dataset.assign(cluster=labels, outlier=NONE_LABEL)

                dataset_id = f"{prefix}+{n_samples}_{n_clusters}_{cluster_std}".replace(".", "_")
                dataset.to_json(output_folder / f"{dataset_id}.json", orient="records", force_ascii=False)

                logger.info("{dataset_id} generated", dataset_id=dataset_id)

                # Noisy background:
                n_noise = 100 if n_samples > 100 else 25

                noise = pd.DataFrame(
                    rng.uniform(
                        [dataset["x"].min(), dataset["y"].min()], [dataset["x"].max(), dataset["y"].max()], (n_noise, 2)
                    ),
                    columns=["x", "y"],
                )
                noise = noise.assign(cluster=NONE_LABEL, outlier=NONE_LABEL)

                dataset = pd.concat([dataset, noise])

                dataset_id = f"{prefix_noise}+{n_samples}_{n_clusters}_{cluster_std}".replace(".", "_")
                dataset.to_json(output_folder / f"{dataset_id}.json", orient="records", force_ascii=False)

                logger.info("{dataset_id} generated", dataset_id=dataset_id)

    total = len(list(output_folder.glob(f"{prefix}+*.json")))
    logger.info("{total} {prefix}+* datasets generated", total=total, prefix=prefix)

    total_noise = len(list(output_folder.glob(f"{prefix_noise}+*.json")))
    logger.info("{total} {prefix}+* datasets generated", total=total_noise, prefix=prefix_noise)


def generate_single_gaussian_blob_outliers(output_folder: Path) -> None:
    prefix = "single_gaussian_blob_outliers"

    for seed in [1, 2, 3]:
        for contamination in [0.001, 0.002, 0.003, 0.004, 0.005, 0.01]:
            for angle in [0, 90, 180]:
                coordinates, labels = generate_data(
                    n_train=1_000, n_features=2, contamination=contamination, train_only=True, random_state=seed
                )

                cluster_labels = np.where(labels == 0, 0, -1)
                outlier_labels = np.where(labels == 1, 0, -1)

                dataset = pd.DataFrame(coordinates, columns=["x", "y"])
                dataset = dataset.assign(cluster=cluster_labels, outlier=outlier_labels)
                dataset = rotate_coordinates_around_center(dataset, angle)

                dataset_id = f"{prefix}+{seed}_{contamination}_{angle}".replace(".", "_")
                dataset.to_json(output_folder / f"{dataset_id}.json", orient="records", force_ascii=False)

                logger.info("{dataset_id} generated", dataset_id=dataset_id)

    total = len(list(output_folder.glob(f"{prefix}+*.json")))
    logger.info("{total} {prefix}+* datasets generated", total=total, prefix=prefix)


def generate_random(output_folder: Path) -> None:
    prefix = "random"

    rng = np.random.default_rng(seed=0)

    # `UserWarning: The balance properties of Sobol' points require n to be a power of 2.`
    # https://en.wikipedia.org/wiki/Power_of_two
    for n_samples in [8, 32, 128, 256]:
        for sampler in [("halton", qmc.Halton), ("sobol", qmc.Sobol)]:
            data = sampler[1](d=2, rng=rng).random(n_samples)

            dataset = pd.DataFrame(data, columns=["x", "y"])
            dataset = dataset.assign(cluster=NONE_LABEL, outlier=NONE_LABEL)

            dataset_id = f"{prefix}+{sampler[0]}_{n_samples}"
            dataset.to_json(output_folder / f"{dataset_id}.json", orient="records", force_ascii=False)

            logger.info("{dataset_id} generated", dataset_id=dataset_id)

    total = len(list(output_folder.glob(f"{prefix}+*.json")))
    logger.info("{total} {prefix}+* datasets generated", total=total, prefix=prefix)


def _generate_linear_relationship(noise_level: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)

    x = np.linspace(0, 10, 100)

    a, b = 2, 1
    y = a * x + b

    gaussian_noise = rng.normal(0, np.std(y) * noise_level, y.size)
    y = y + gaussian_noise

    dataset = pd.DataFrame({"x": x, "y": y})

    return dataset.assign(cluster=NONE_LABEL, outlier=NONE_LABEL)


def _generate_exponential_relationship(noise_level: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)

    x = np.linspace(0, 10, 100)

    a, b = 2, 1
    y = a * np.exp(b * x)

    gaussian_noise = rng.normal(0, np.std(y) * noise_level, y.size)
    y = y + gaussian_noise

    dataset = pd.DataFrame({"x": x, "y": y})

    return dataset.assign(cluster=NONE_LABEL, outlier=NONE_LABEL)


def _generate_quadratic_relationship(noise_level: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)

    x = np.linspace(-5, 5, 100)

    a, b, c = 2, 1, 2
    y = a * x**2 + b * x + c

    gaussian_noise = rng.normal(0, np.std(y) * noise_level, y.size)
    y = y + gaussian_noise

    dataset = pd.DataFrame({"x": x, "y": y})

    return dataset.assign(cluster=NONE_LABEL, outlier=NONE_LABEL)


def generate_relationship(output_folder: Path) -> None:
    prefix = "relationship"

    for relationship in [
        ("linear", _generate_linear_relationship),
        ("exponential", _generate_exponential_relationship),
        ("quadratic", _generate_quadratic_relationship),
    ]:
        for noise_level in [0.0, 0.1, 0.2]:
            dataset = relationship[1](noise_level)

            dataset_id = f"{prefix}+{relationship[0]}_{noise_level}".replace(".", "_")
            dataset.to_json(output_folder / f"{dataset_id}.json", orient="records", force_ascii=False)

            logger.info("{dataset_id} generated", dataset_id=dataset_id)

    total = len(list(output_folder.glob(f"{prefix}+*.json")))
    logger.info("{total} {prefix}+* datasets generated", total=total, prefix=prefix)


def _generate_diamond(centroids: list[list[float]], r: float) -> list[Polygon]:
    polygons: list[Polygon] = []

    for centroid in centroids:
        vertices: list[tuple[float, float]] = []

        for i in range(4):
            angle = math.radians(i * 90)

            x = centroid[0] + r * math.cos(angle)
            y = centroid[1] + r * math.sin(angle)

            vertices.append((x, y))

        polygons.append(Polygon(vertices))

    return polygons


def _generate_hexagon(centroids: list[list[float]], r: float) -> list[Polygon]:
    polygons: list[Polygon] = []

    for centroid in centroids:
        vertices: list[tuple[float, float]] = []

        for i in range(6):
            angle = math.radians(90 + i * 60)

            x = centroid[0] + r * math.cos(angle)
            y = centroid[1] + r * math.sin(angle)

            vertices.append((x, y))

        polygons.append(Polygon(vertices))

    return polygons


def _generate_square(centroids: list[list[float]], r: float) -> list[Polygon]:
    polygons: list[Polygon] = []

    for centroid in centroids:
        vertices: list[tuple[float, float]] = []

        for i in range(4):
            angle = math.radians(i * 90 + 45)

            x = centroid[0] + r * math.cos(angle)
            y = centroid[1] + r * math.sin(angle)

            vertices.append((x, y))

        polygons.append(Polygon(vertices))

    return polygons


def _generate_star(centroids: list[list[float]], r: float) -> list[Polygon]:
    polygons: list[Polygon] = []

    for centroid in centroids:
        vertices: list[tuple[float, float]] = []

        n_points = 5
        inner_radius = r * 0.4

        for i in range(n_points * 2):
            angle = math.radians(i * 36 + 90)
            current_radius = r if i % 2 == 0 else inner_radius

            x = centroid[0] + current_radius * math.cos(angle)
            y = centroid[1] + current_radius * math.sin(angle)

            vertices.append((x, y))

        polygons.append(Polygon(vertices))

    return polygons


def _generate_triangle_down(centroids: list[list[float]], r: float) -> list[Polygon]:
    polygons: list[Polygon] = []

    for centroid in centroids:
        vertices: list[tuple[float, float]] = []

        for i in range(3):
            angle = math.radians(i * 120 - 90)

            x = centroid[0] + r * math.cos(angle)
            y = centroid[1] + r * math.sin(angle)

            vertices.append((x, y))

        polygons.append(Polygon(vertices))

    return polygons


def _generate_triangle_right(centroids: list[list[float]], r: float) -> list[Polygon]:
    polygons: list[Polygon] = []

    for centroid in centroids:
        vertices: list[tuple[float, float]] = [
            (centroid[0] - r, centroid[1] - r),
            (centroid[0] + r, centroid[1] - r),
            (centroid[0] - r, centroid[1] + r),
        ]

        polygons.append(Polygon(vertices))

    return polygons


def generate_shapes(output_folder: Path) -> None:
    prefix = "shapes"

    dimensions = 2
    n_points_per_cluster = 200

    # Poisson disk sampling radius or circle diameter:
    radius = 1
    circle_radius = radius / 2

    min_n_clusters = 2
    max_n_clusters = 6

    for shape in [
        ("diamond", _generate_diamond),
        ("hexagon", _generate_hexagon),
        ("square", _generate_square),
        ("star", _generate_star),
        ("triangle_down", _generate_triangle_down),
        ("triangle_right", _generate_triangle_right),
    ]:
        for seed in [0, 1, 42]:
            rng = np.random.default_rng(seed=seed)

            for n_clusters in range(min_n_clusters, max_n_clusters + 1):
                engine = qmc.PoissonDisk(
                    d=dimensions,
                    radius=radius,
                    rng=rng,
                    l_bounds=np.zeros(dimensions),
                    u_bounds=np.full(dimensions, radius * n_clusters),
                )
                centroids = engine.random(n_clusters).tolist()

                for r in [circle_radius, radius]:
                    polygons = shape[1](centroids, r)
                    data: list[tuple[float, float, int]] = []

                    for index, polygon in enumerate(polygons):
                        polygon_minx, polygon_miny, polygon_maxx, polygon_maxy = polygon.bounds

                        while len(data) < n_points_per_cluster * (index + 1):
                            point = Point(
                                rng.uniform(polygon_minx, polygon_maxx), rng.uniform(polygon_miny, polygon_maxy)
                            )

                            if polygon.contains(point):
                                data.append((point.x, point.y, index))

                    dataset = pd.DataFrame(data, columns=["x", "y", "cluster"])
                    dataset = dataset.assign(outlier=NONE_LABEL)

                    dataset_id = f"{prefix}+{shape[0]}_{seed}_{n_clusters}_{r}".replace(".", "_")
                    dataset.to_json(output_folder / f"{dataset_id}.json", orient="records", force_ascii=False)

                    logger.info("{dataset_id} generated", dataset_id=dataset_id)

    total = len(list(output_folder.glob(f"{prefix}+*.json")))
    logger.info("{total} {prefix}+* datasets generated", total=total, prefix=prefix)


if __name__ == "__main__":
    ensure_clean_dir(DATASETS)

    generate_gaussian_blobs(DATASETS)
    generate_single_gaussian_blob_outliers(DATASETS)
    generate_random(DATASETS)
    generate_relationship(DATASETS)
    generate_shapes(DATASETS)

    all_datasets = list(DATASETS.glob("*.json"))

    logger.info("{total} datasets generated", total=len(all_datasets))
    logger.info(
        "Dataset breakdown: {breakdown}",
        breakdown=Counter(dataset.stem.split("+", maxsplit=1)[0] for dataset in all_datasets),
    )
