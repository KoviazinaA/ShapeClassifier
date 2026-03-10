"""
Image generation module.

Generates synthetic images of triangles and circles
with controllable variation (size, position, color, noise).

Drawing approach adapted from KoviazinaA/ShapeClassifier:
  - Triangles: 3 random non-collinear points filled via matplotlib.
  - Circles:   parametric equation with random centre/radius.
Both shapes use random fill + edge colours on a white background.
"""

import io
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


LABELS = ("triangle", "circle")

# Coordinate space used by matplotlib before rasterising to pixels.
_COORD_RANGE = (0.0, 5.0)


def _random_rgb() -> tuple[float, float, float]:
    """Return a random (R, G, B) tuple in [0, 1]³."""
    return (random.random(), random.random(), random.random())


def _generate_3_points(
    x_range: tuple[float, float] = _COORD_RANGE,
    y_range: tuple[float, float] = _COORD_RANGE,
) -> np.ndarray:
    """
    Generate 3 random non-collinear points.

    Returns:
        Shape (2, 3) array — row 0 is x coords, row 1 is y coords.
    """
    while True:
        points = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(3)]
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        # Determinant-based area; > 0 guarantees non-collinear vertices.
        area = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
        if area > 0:
            return np.array(points).T


def _fig_to_array(fig: plt.Figure, img_size: int) -> np.ndarray:
    """
    Rasterise a matplotlib figure to a uint8 RGB numpy array of shape
    (img_size, img_size, 3).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((img_size, img_size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def generate_triangle(
    img_size: int = 64,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Draw a random triangle and return it as a uint8 RGB array.

    Args:
        img_size: Output image size in pixels (square).
        noise_std: Std-dev of Gaussian pixel noise added after rendering.

    Returns:
        np.ndarray of shape (img_size, img_size, 3), dtype uint8.
    """
    x, y = _generate_3_points()

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.fill(x, y, color=_random_rgb(), edgecolor=_random_rgb())
    ax.set_xlim(*_COORD_RANGE)
    ax.set_ylim(*_COORD_RANGE)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    arr = _fig_to_array(fig, img_size)
    plt.close(fig)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return arr


def generate_circle(
    img_size: int = 64,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Draw a random circle and return it as a uint8 RGB array.

    Args:
        img_size: Output image size in pixels (square).
        noise_std: Std-dev of Gaussian pixel noise added after rendering.

    Returns:
        np.ndarray of shape (img_size, img_size, 3), dtype uint8.
    """
    cx = random.uniform(1.5, 3.5)
    cy = random.uniform(1.5, 3.5)
    r = random.uniform(0.5, 1.5)

    theta = np.linspace(0, 2 * np.pi, 300)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.fill(x, y, color=_random_rgb(), edgecolor=_random_rgb())
    ax.set_xlim(*_COORD_RANGE)
    ax.set_ylim(*_COORD_RANGE)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    arr = _fig_to_array(fig, img_size)
    plt.close(fig)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return arr


def generate_dataset(
    output_dir: str | Path,
    n_samples: int,
    img_size: int = 64,
    noise_std: float = 10.0,
    seed: int = 42,
) -> None:
    """
    Generate `n_samples` images per class and save to output_dir/<label>/.

    Folder structure created:
        output_dir/triangle/triangle_0000.png
        output_dir/triangle/triangle_0001.png
        ...
        output_dir/circle/circle_0000.png
        ...

    Args:
        output_dir: Root directory; sub-folders per label are created automatically.
        n_samples: Number of images to create for each class.
        img_size: Width and height of each square image in pixels.
        noise_std: Standard deviation of Gaussian noise added to each image.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    generators = {
        "triangle": generate_triangle,
        "circle": generate_circle,
    }

    for label, gen_fn in generators.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_samples):
            arr = gen_fn(img_size=img_size, noise_std=noise_std)
            Image.fromarray(arr).save(label_dir / f"{label}_{i:04d}.png")
        print(f"  {label}: {n_samples} images saved to {label_dir}")

    print(f"Done. Total images: {len(LABELS) * n_samples}")


if __name__ == "__main__":
    generate_dataset(output_dir="data/raw", n_samples=1000)
