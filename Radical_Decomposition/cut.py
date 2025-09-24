import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_rgb_and_gray(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return the RGB numpy array and its grayscale version for the given image path."""

    image = Image.open(path)
    if image.mode == "RGBA":
        # Replace transparent pixels with white so the mask finds actual content.
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        white_bg.paste(image, mask=image.split()[3])
        rgb_array = np.array(white_bg)
        gray_array = np.array(white_bg.convert("L"))
    else:
        rgb_image = image.convert("RGB")
        rgb_array = np.array(rgb_image)
        gray_array = np.array(rgb_image.convert("L"))
    return rgb_array, gray_array


def bounding_rect_from_contours(gray: np.ndarray, threshold: int) -> Optional[Tuple[int, int, int, int]]:
    """Compute a slightly padded bounding rectangle around the non-background area."""

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    if len(contours) > 1:
        merged = contours[0]
        for contour in contours[1:]:
            merged = np.vstack((merged, contour))
        x, y, w, h = cv2.boundingRect(merged)
    else:
        x, y, w, h = cv2.boundingRect(contours[0])

    # Nudge the rectangle outward to avoid clipping pixels on the edge.
    if x + w < gray.shape[1]:
        w += 1
    if x + w < gray.shape[1]:
        w += 1
    if x > 1:
        x -= 1
        w += 1
    if x > 1:
        x -= 1
        w += 1
    if y + h < gray.shape[0]:
        h += 1
    if y + h < gray.shape[0]:
        h += 1
    if y > 1:
        y -= 1
        h += 1
    if y > 1:
        y -= 1
        h += 1

    return x, y, w, h


def crop_image(path: Path, threshold: int) -> bool:
    """Crop the image in-place using a threshold to remove background; return True if cropped."""

    rgb, gray = load_rgb_and_gray(path)
    rect = bounding_rect_from_contours(gray, threshold)
    if rect is None:
        print(path)
        return False

    x, y, w, h = rect
    cut = rgb[y : y + h, x : x + w]
    opencv_image_rgb = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
    Image.fromarray(opencv_image_rgb).save(path)
    return True


def resolve_image_path(entry: dict, dataset_path: Path) -> Path:
    """Resolve image path relative to the dataset file when needed."""

    raw_path = entry.get("path")
    if raw_path is None:
        raise KeyError("Dataset entry missing 'path' field")
    image_path = Path(raw_path)
    # if not image_path.is_absolute():
    #     image_path = (dataset_path.parent / image_path).resolve()
    return image_path


def process_dataset(dataset_path: Path, threshold: int) -> None:
    """Load dataset entries and crop each referenced image."""

    with dataset_path.open("r", encoding="utf8") as handle:
        data = json.load(handle)

    for entry in tqdm(data, desc="Cropping"):
        image_path = resolve_image_path(entry, dataset_path)
        crop_image(image_path, threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop puzzle piece images by removing blank backgrounds.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("./output/Decomposition_Dataset.json"),
        help="Path to the JSON dataset file (e.g., output/Decomposition_Dataset.json)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=160,
        help="Grayscale threshold for separating background from foreground (default: 160)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(args.dataset, args.threshold)


if __name__ == "__main__":
    main()
