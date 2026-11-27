"""Image preprocessing and feature extraction with OpenCV."""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

DEFAULT_IMAGE_SIZE = (32, 32)
DATASET_DIR = Path("data/clean_dataset")
OUTPUT_DIR = Path("traditional") / "outputs" / "preprocessed_images"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def apply_clahe(
    image: np.ndarray, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply CLAHE to grayscale or the value channel of an HSV image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a grayscale or 3-channel HSV image.")

    hsv_image = image.copy()
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])
    return hsv_image


def apply_gaussian_smoothing(
    image: np.ndarray, kernel_size: tuple[int, int] = (5, 5), sigma: float = 0
) -> np.ndarray:
    """Blur grayscale or the value channel of an HSV image."""
    if image.ndim == 2:
        return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a grayscale or 3-channel HSV image.")

    smoothed_image = image.copy()
    smoothed_image[:, :, 2] = cv2.GaussianBlur(
        image[:, :, 2], kernel_size, sigmaX=sigma
    )
    return smoothed_image


def resize_image(
    image: np.ndarray, target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Resize to (width, height)."""
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Expected a non-empty grayscale or color image.")
    if len(target_size) != 2 or any(
        not isinstance(size, int) or size <= 0 for size in target_size
    ):
        raise ValueError("target_size must contain two positive integers.")

    width, height = image.shape[1], image.shape[0]
    target_width, target_height = target_size
    if (width, height) == target_size:
        return image

    shrinking = target_width <= width and target_height <= height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(image, target_size, interpolation=interpolation)


def extract_canny_edges(image: np.ndarray) -> np.ndarray:
    """Detect edges in the CLAHE-enhanced HSV value channel."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    resized_image = resize_image(image)
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    hsv_image = apply_clahe(hsv_image, tile_grid_size=(2, 2))
    hsv_image = apply_gaussian_smoothing(
        hsv_image, kernel_size=(3, 3), sigma=0.8
    )
    value = hsv_image[:, :, 2]
    return cv2.Canny(value, threshold1=25, threshold2=75, L2gradient=True)

"""======================================================================================"""

"""Every function below forms something in outputs folder. 
    preprocess_image() = makes edge detected map
    extracT_hsv_histogram() = makes HSV histogram, needs tweaking?
    save_preprocessing_comparison() = makes the figure showing the image in all 4 forms
    preprocess_dataset() = processes the entire /clean_dataset/ directory, and outputs it in traditional/outputs/preprocessed_images,
    sorted by folder (i hope)
    main() = calls above functions depending on arguments when running program
"""
"""   
    Example test if u wanna try: 
    'python traditional\preprocess.py data\clean_dataset\images\19\image_013150.png --figure' 

    Gets a sorta funky image from the clean_dataset and processes it, outputting the preprocessed image, and puts it
    in the directory traditional/outputs/preprocessed_images.  """

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Return a 32x32 Canny edge map."""
    return extract_canny_edges(image)


def extract_hsv_histogram(
    image: np.ndarray, bins: tuple[int, int, int] = (8, 8, 8)
) -> np.ndarray:
    """Return a flattened, normalized 3D HSV histogram."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")
    if len(bins) != 3 or any(not isinstance(count, int) or count <= 0 for count in bins):
        raise ValueError("bins must contain three positive integers.")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv_image],
        [0, 1, 2],
        None,
        list(bins),
        [0, 180, 0, 256, 0, 256],
    )
    return cv2.normalize(histogram, None, norm_type=cv2.NORM_L1).flatten()


def save_preprocessing_comparison(image: np.ndarray, output_path: Path) -> None:
    """Save the four processing stages in a single row."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe_image = apply_clahe(hsv_image)
    smoothed_image = apply_gaussian_smoothing(clahe_image)
    stages = [
        ("Original", image),
        ("CLAHE", cv2.cvtColor(clahe_image, cv2.COLOR_HSV2BGR)),
        ("CLAHE + Gaussian smoothing", cv2.cvtColor(smoothed_image, cv2.COLOR_HSV2BGR)),
        ("Canny edge map", extract_canny_edges(image)),
    ]
    panels = [(title, resize_image(panel)) for title, panel in stages]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    title_sizes = [
        cv2.getTextSize(title, font, font_scale, thickness)[0]
        for title, _ in panels
    ]
    padding = 8
    gap = 12
    image_height, image_width = panels[0][1].shape[:2]
    column_width = max(
        image_width, max(width for width, _ in title_sizes)
    ) + 2 * padding
    title_height = max(size[1] for size in title_sizes) + padding
    row_height = title_height + image_height + padding
    canvas = np.full(
        (row_height + 2 * gap, 4 * column_width + 5 * gap, 3),
        255,
        dtype=np.uint8,
    )

    for index, ((title, panel), title_size) in enumerate(zip(panels, title_sizes)):
        x = gap + index * (column_width + gap)
        y = gap
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        title_x = x + (column_width - title_size[0]) // 2
        cv2.putText(
            canvas,
            title,
            (title_x, y + title_height - padding),
            font,
            font_scale,
            (30, 30, 30),
            thickness,
            cv2.LINE_AA,
        )
        image_x = x + (column_width - image_width) // 2
        image_y = y + title_height
        canvas[
            image_y:image_y + image_height,
            image_x:image_x + image_width,
        ] = panel

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write comparison image: {output_path}")


def preprocess_dataset(dataset_dir: Path, output_dir: Path) -> None:
    """Preprocess dataset images and keep their class folders and labels."""
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {images_dir}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Dataset labels not found: {labels_path}")

    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read image: {image_path}")

        output_path = output_dir / "images" / image_path.relative_to(images_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), preprocess_image(image)):
            raise OSError(f"Could not write image: {output_path}")
        if index % 1000 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)} images")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_path, output_dir / labels_path.name)
    print(f"Preprocessed dataset saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a single image or the clean dataset."
    )
    parser.add_argument("input", nargs="?", type=Path, help="Path to the input image.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help=(
            "Output path (defaults to "
            "traditional\\outputs\\preprocessed_images\\<input>_preprocessed)."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Preprocess data\\clean_dataset into traditional\\outputs\\preprocessed_images.",
    )
    parser.add_argument(
        "--figure",
        nargs="?",
        const="auto",
        metavar="PATH",
        help="Save a comparison image; default name uses the input filename.",
    )
    args = parser.parse_args()

    if args.dataset:
        if args.input or args.output or args.figure:
            parser.error("--dataset cannot be combined with image paths or --figure")
        preprocess_dataset(DATASET_DIR, OUTPUT_DIR)
        return
    if args.input is None:
        parser.error("an input image is required unless --dataset is used")

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Could not read input image: {args.input}")

    processed_image = preprocess_image(image)
    output_path = args.output or OUTPUT_DIR / (
        f"{args.input.stem}_preprocessed{args.input.suffix}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), processed_image):
        parser.error(f"Could not write output image: {output_path}")
    print(f"Processed image saved to: {output_path}")

    if args.figure is not None:
        figure_path = (
            OUTPUT_DIR / f"preprocessing_comparison_{args.input.stem}.png"
            if args.figure == "auto"
            else Path(args.figure)
        )
        save_preprocessing_comparison(image, figure_path)
        print(f"Comparison image saved to: {figure_path}")


if __name__ == "__main__":
    main()