"""
traditional/joe.py

Traditional Computer Vision pipeline for GTSRB Traffic Sign Classification.
Handles HOG feature extraction, optional integration with Soph's preprocessing
and HSV colour histograms, PCA dimensionality reduction, and classical ML
classification (LinearSVC and Random Forest).
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# from traditional.soph import preprocess_image, extract_hsv_hist


# ===========================================================================
# 1. FEATURE EXTRACTION FUNCTIONS
# ===========================================================================

def extract_hog(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a single 32x32 BGR image array to grayscale and extract its
    1D Histogram of Oriented Gradients (HOG) feature vector.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),   # 32x32 image -> 4x4 grid of cells
        cells_per_block=(2, 2),   # 3x3 overlapping blocks -> 324 features
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def extract_features_batch(
    images: list[np.ndarray] | np.ndarray,
    use_preprocessing: bool = False,
    use_hsv: bool = False,
) -> np.ndarray:
    """
    Extract a 2D feature matrix (num_images, num_features) from a collection
    of BGR image arrays. Toggles allow clean ablation testing with Soph's code.
    """
    feature_list = []

    for img in images:
        processed_img = img

        # Hook 1: Soph's traditional preprocessing (CLAHE + Gaussian smoothing)
        if use_preprocessing:
            # Replace this raise statement once Soph's function is imported:
            # processed_img = preprocess_image(img)
            raise NotImplementedError("Import preprocess_image from Soph's module first.")

        # Core feature: HOG (shape and edge orientations)
        hog_feats = extract_hog(processed_img)

        # Hook 2: Soph's HSV colour histogram
        if use_hsv:
            # Replace this raise statement once Soph's function is imported:
            # hsv_feats = extract_hsv_hist(processed_img)
            # combined_feats = np.concatenate([hog_feats, hsv_feats])
            raise NotImplementedError("Import extract_hsv_hist from Soph's module first.")
        else:
            combined_feats = hog_feats

        feature_list.append(combined_feats)

    return np.array(feature_list, dtype=np.float32)


# ===========================================================================
# 2. MODEL PIPELINE BUILDER
# ===========================================================================

def build_pipeline(model_type: str = "linear_svc") -> Pipeline:
    """
    Build a scikit-learn Pipeline that chains Standardization, PCA, and a
    classical classifier to prevent data leakage and keep training clean.
    """
    if model_type == "linear_svc":
        classifier = LinearSVC(
            random_state=42,
            max_iter=2000,
            dual="auto",
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,  # Uses all available CPU cores for faster training
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Use 'linear_svc' or 'random_forest'.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("clf", classifier),
    ])
    return pipeline


# ===========================================================================
# 3. EXTERNAL INTERFACE (FOR AADI'S TESTING SUITE)
# ===========================================================================

def train(
    X_train_imgs: list[np.ndarray] | np.ndarray,
    y_train: np.ndarray,
    model_type: str = "linear_svc",
    use_preprocessing: bool = False,
    use_hsv: bool = False,
) -> Pipeline:
    """
    Extract features from training images, fit the classical pipeline,
    and return the trained model object.
    """
    X_train_feats = extract_features_batch(
        X_train_imgs,
        use_preprocessing=use_preprocessing,
        use_hsv=use_hsv,
    )
    model = build_pipeline(model_type=model_type)
    model.fit(X_train_feats, y_train)
    return model


def classify(
    model: Pipeline,
    X_test_imgs: list[np.ndarray] | np.ndarray,
    use_preprocessing: bool = False,
    use_hsv: bool = False,
) -> np.ndarray:
    """
    Extract features from unseen test images using the same settings as
    training, and return predicted ClassIds.
    """
    X_test_feats = extract_features_batch(
        X_test_imgs,
        use_preprocessing=use_preprocessing,
        use_hsv=use_hsv,
    )
    return model.predict(X_test_feats)


# ===========================================================================
# 4. LOCAL HELPERS (FOR INDEPENDENT TESTING & CSV EXPORT)
# ===========================================================================

def load_local_data(csv_path: Path, data_dir: Path):
    """
    Read labels.csv and load all corresponding 32x32 BGR images into memory.
    """
    df = pd.read_csv(csv_path)
    images, labels, paths = [], [], []

    for row in df.itertuples(index=False):
        img_path = data_dir / row.Path
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        images.append(img)
        labels.append(row.ClassId)
        paths.append(row.Path)

    return images, np.array(labels), np.array(paths)


def save_predictions(
    paths: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_csv: Path,
) -> None:
    """
    Save test image paths alongside ground-truth and predicted labels to CSV.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        "Path": paths,
        "true_label": y_true,
        "pred_label": y_pred,
    })
    results_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")


# ===========================================================================
# 5. LOCAL EXECUTION BLOCK
# ===========================================================================

if __name__ == "__main__":
    # Resolve paths relative to CompVisCW2/traditional/joe.py
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data" / "clean_dataset"
    LABELS_CSV = DATA_DIR / "labels.csv"
    OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

    print("Loading clean_dataset images into memory...")
    images, y, paths = load_local_data(LABELS_CSV, DATA_DIR)
    print(f"Loaded {len(images)} images.")

    # Local 80/20 stratified split for independent testing
    X_train_imgs, X_test_imgs, y_train, y_test, _, test_paths = train_test_split(
        images, y, paths, test_size=0.2, random_state=42, stratify=y
    )

    # Run and compare both classical classifiers on raw HOG features
    for model_name in ["linear_svc", "random_forest"]:
        print(f"\n--- Running {model_name} (Raw HOG baseline) ---")
        fitted_model = train(
            X_train_imgs,
            y_train,
            model_type=model_name,
            use_preprocessing=False,
            use_hsv=False,
        )
        y_pred = classify(
            fitted_model,
            X_test_imgs,
            use_preprocessing=False,
            use_hsv=False,
        )

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        print(f"{model_name} -> Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")

        save_predictions(
            test_paths,
            y_test,
            y_pred,
            OUTPUT_DIR / f"{model_name}_hog_baseline.csv",
        )