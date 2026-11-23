from pathlib import Path
import zipfile
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import cv2
from tqdm import tqdm

ZIP_NAME = "archive.zip"
EXTRACTED_FOLDER_NAME = "raw_dataset"
OUTPUT_FOLDER_NAME = "clean_dataset"
TARGET_SIZE = (32, 32)

ROI_X1 = "Roi.X1"
ROI_Y1 = "Roi.Y1"
ROI_X2 = "Roi.X2"
ROI_Y2 = "Roi.Y2"
FILE_NAME = "Path"
CLASS_LABEL = "ClassId"


def main():
    # Zip extraction function
    print("Starting zip extraction")
    zip_path = Path(ZIP_NAME)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file '{ZIP_NAME}' not found.")
    if Path(EXTRACTED_FOLDER_NAME).exists():
        print(f"Folder '{EXTRACTED_FOLDER_NAME}' already exists. Skipping extraction.")
    else:
        with zipfile.ZipFile(ZIP_NAME, "r") as zf:
            members = zf.infolist()
            for member in tqdm(members, desc="Extracting archive", unit="file"):
                zf.extract(member, EXTRACTED_FOLDER_NAME)
        print("Finished zip extraction")

    # Create the output folder if it doesn't exist
    print("Creating output folder if it doesn't exist")
    output_path = Path(OUTPUT_FOLDER_NAME)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Extract the CSV files and load them and find the elligible images
    print("Starting CSV filtering")
    train_data = pd.read_csv("raw_dataset/train.csv")
    test_data = pd.read_csv("raw_dataset/test.csv")
    filtered_data = []
    for df in [train_data, test_data]:
        widths = df[ROI_X2] - df[ROI_X1]
        heights = df[ROI_Y2] - df[ROI_Y1]
        mask = (widths >= TARGET_SIZE[0]) & (heights >= TARGET_SIZE[1])
        filtered = df.loc[mask].copy()
        filtered_data.append(filtered)
    clean_data = pd.concat(filtered_data, ignore_index=True)
    print("Finished CSV filtering")

    # Start cleaning the images using multiprocessing
    print("Starting image cleaning")
    tasks = list(zip(
        clean_data.index,
        clean_data[FILE_NAME],
        clean_data[ROI_X1],
        clean_data[ROI_Y1],
        clean_data[ROI_X2],
        clean_data[ROI_Y2],
        clean_data[CLASS_LABEL]
    ))
    with ProcessPoolExecutor() as pool:
        results = list(
            tqdm(
                pool.map(clean_image, tasks, chunksize=128),
                total=len(tasks),
                desc="Cleaning images",
                unit="img",
            )
        )
    print(f"Finished image cleaning: {sum(results)}/{len(results)}")

    # Create a new CSV file with the cleaned data
    print("Creating cleaned CSV file")
    new_paths = [f"images/{cls}/image_{i:06d}.png" for i, cls in zip(clean_data.index, clean_data[CLASS_LABEL])]
    labels_df = pd.DataFrame({FILE_NAME: new_paths, CLASS_LABEL: clean_data[CLASS_LABEL]})
    labels_df = labels_df[results].reset_index(drop=True)
    labels_df.to_csv(output_path / "labels.csv", index=False)
    print("Finished creating cleaned CSV file")

    print("\nAll tasks completed successfully.")


# Function to clean an individual image and save it
def clean_image(args): 
    # Set up variables
    i, path, x1, y1, x2, y2, class_label = args
    in_file = Path(EXTRACTED_FOLDER_NAME) / path
    out_file = Path(OUTPUT_FOLDER_NAME) / "images" / str(class_label) / f"image_{i:06d}.png"
    cv2.setNumThreads(1)
    # Open the image
    img = cv2.imread(str(in_file))
    # Crop the image using the ROI coordinates
    cropped = img[y1:y2, x1:x2]
    # Resize the cropped image to the target size
    if cropped.shape[:2] == TARGET_SIZE[::-1]:
        resized = cropped
    else:
        resized = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    # Save the cleaned image to the output folder
    out_file.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_file), resized))

if __name__ == "__main__":
    main()