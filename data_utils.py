import random
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
S_DIR = BASE_DIR / "s_images"
ANTI_S_DIR = BASE_DIR / "anti_s_images"


def _paths_by_pair_id(directory, prefix):
    paths = {}
    for path in directory.glob(f"{prefix}*.png"):
        pair_id = int(path.stem.removeprefix(prefix))
        paths[pair_id] = path
    return paths


def get_train_test_paths(test_pair_count=500, seed=42):
    """Reserve matching S/mirror pairs for testing to prevent pair leakage."""
    s_paths = _paths_by_pair_id(S_DIR, "s_image_")
    anti_s_paths = _paths_by_pair_id(ANTI_S_DIR, "anti_s_image_")
    pair_ids = sorted(s_paths.keys() & anti_s_paths.keys())
    if len(pair_ids) <= test_pair_count:
        raise ValueError(
            f"Need more than {test_pair_count} matched image pairs; found {len(pair_ids)}."
        )

    test_ids = set(random.Random(seed).sample(pair_ids, test_pair_count))
    train_s = [s_paths[i] for i in sorted(s_paths.keys() - test_ids)]
    train_anti_s = [anti_s_paths[i] for i in sorted(anti_s_paths.keys() - test_ids)]
    test_s = [s_paths[i] for i in sorted(test_ids)]
    test_anti_s = [anti_s_paths[i] for i in sorted(test_ids)]
    return train_s, train_anti_s, test_s, test_anti_s


def load_images(image_paths):
    images = np.empty((len(image_paths), 35, 35, 1), dtype=np.float32)
    for index, path in enumerate(image_paths):
        pixels = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        images[index, ..., 0] = 1.0 - pixels
    return images


def chirality_pseudo_labels(images):
    """Create rotation-invariant reflection labels from canonical image moments."""
    y_grid, x_grid = np.indices(images.shape[1:3], dtype=np.float64)
    labels = np.empty(len(images), dtype=np.float32)

    for index, image in enumerate(images[..., 0]):
        weights = image.astype(np.float64)
        mass = weights.sum()
        if mass <= 0:
            raise ValueError(f"Image at batch index {index} contains no foreground pixels.")

        x = x_grid - np.sum(weights * x_grid) / mass
        y = y_grid - np.sum(weights * y_grid) / mass
        covariance = np.array(
            [
                [np.sum(weights * x * x), np.sum(weights * x * y)],
                [np.sum(weights * x * y), np.sum(weights * y * y)],
            ]
        ) / mass
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_axis = eigenvectors[:, np.argmax(eigenvalues)]
        u = x * major_axis[0] + y * major_axis[1]

        if np.sum(weights * u**3) < 0:
            major_axis = -major_axis
            u = -u

        minor_axis = np.array([-major_axis[1], major_axis[0]])
        v = x * minor_axis[0] + y * minor_axis[1]
        std_u = np.sqrt(np.sum(weights * u * u) / mass)
        std_v = np.sqrt(np.sum(weights * v * v) / mass)
        if std_u <= 0 or std_v <= 0:
            raise ValueError(f"Image at batch index {index} has degenerate spatial moments.")

        signed_moment = np.sum(weights * (u / std_u) ** 3 * (v / std_v)) / mass
        if np.isclose(signed_moment, 0.0):
            raise ValueError(f"Image at batch index {index} has ambiguous chirality.")
        labels[index] = float(signed_moment > 0)

    return labels
